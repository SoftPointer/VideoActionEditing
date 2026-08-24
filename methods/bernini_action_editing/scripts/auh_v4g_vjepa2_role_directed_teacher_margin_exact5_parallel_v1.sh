#!/usr/bin/env bash
# SEALED executable controller for the exact-five v4-G role-directed
# teacher-margin diagnostic.  It only consumes five fixed live AUH holders;
# it never submits, cancels, resizes, falls back, or chooses another node.
#
# External callers provide only sealed source/input roots, a fresh run root,
# and this detached controller's SHA.  Inner, barrier, and fold receipt SHAs
# are captured and verified by this controller and are never caller inputs.

set -Eeuo pipefail
umask 077

readonly tag=v4g-vjepa2-role-directed-teacher-margin-exact5-parallel-v1
readonly release_sealed=true
readonly python_pin_sealed=true
readonly controller_contract_complete=true
readonly placeholder_sha256=0000000000000000000000000000000000000000000000000000000000000000

readonly expected_release_manifest_sha256=d1e0c42904057e14d47c87e746c32db375ba6ee6b006f813ea91cdb2daae4882
readonly expected_release_manifest_digest=91adc268b8b86f083c36aa043e5c6f10956e720b03a3b5ebf1022b3775ac46a4
readonly expected_release_tree_sha256=e4e158e064ceb181345673c86a8fb275436ddd25edc74a3e7ef8f1c31d4f16ff
readonly expected_runtime_sha256=38b2cbecaf022e203ccf09e6808661013f4f23dee0d02ffa1756e24d0c167cf9
readonly expected_runtime_test_sha256=7fe6b42208f77171f99d44d5a9fc9eae58c3bb2d4663ca016e9b154a4d3c4996
readonly expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a

readonly expected_feature_authority_sha256=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly expected_v2_runtime_sha256=46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
readonly expected_v4a_runtime_sha256=e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
readonly expected_extractor_sha256=720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc
readonly expected_v4c_runtime_sha256=d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef
readonly expected_v4d_runtime_sha256=20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc
readonly expected_v4e_burned_runtime_sha256=4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a
readonly expected_v4f_runtime_sha256=97cd77e64a4dfaf3036e6c50a5b85060fd616f87371e5d967e69db1170466d74

readonly expected_feature_receipt_sha256=895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a
readonly expected_v4a_receipt_sha256=568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2
readonly expected_v4c_frontier_receipt_sha256=8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9
readonly expected_v4d_receipt_sha256=53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f

readonly expected_inner_schema=semantic-anchor-vjepa2-role-directed-teacher-margin-inner-receipt-v4g
readonly expected_fold_schema=semantic-anchor-vjepa2-role-directed-teacher-margin-fold-receipt-v4g
readonly expected_barrier_schema=semantic-anchor-vjepa2-role-directed-teacher-margin-global-inner-barrier-v4g
readonly expected_aggregate_schema=semantic-anchor-vjepa2-role-directed-teacher-margin-exact5-receipt-v4g
readonly expected_checkpoint_schema=semantic-anchor-vjepa2-role-directed-teacher-margin-checkpoint-v4g
readonly expected_inner_pass_status=V4G_FIXED1200_INNER_PASS_OOF_UNREAD
readonly expected_inner_no_go_status=V4G_FIXED1200_INNER_NO_GO_ALL_OOF_UNREAD
readonly expected_barrier_pass_status=V4G_EXACT5_INNER_BARRIER_PASS_OOF_UNREAD
readonly expected_aggregate_status=V4G_ROLE_DIRECTED_TEACHER_MARGIN_KNOWN_EXPOSED_DEVELOPMENT

readonly -a worker_folds=(0 1 2 3 4)
readonly -a worker_jobs=(143808 143812 143811 143808 143808)
readonly -a worker_nodes=(
  auh7-1b-gpu-315
  auh7-1b-gpu-293
  auh7-1b-gpu-306
  auh7-1b-gpu-233
  auh7-1b-gpu-268
)
readonly worker_cpus=8
readonly worker_memory=12G
readonly preflight_job=143811
readonly preflight_node=auh7-1b-gpu-306
readonly preflight_cpus=4
readonly preflight_memory=4G
readonly cpu_job=143811
readonly cpu_node=auh7-1b-gpu-306
readonly cpu_cpus=8
readonly cpu_memory=12G
readonly barrier_worker_index=2

readonly feature_authority_relative=methods/bernini_action_editing/semantic_action_cvae_canary_v1.py
readonly v2_runtime_relative=methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
readonly v4a_runtime_relative=methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
readonly extractor_relative=methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py
readonly v4c_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py
readonly v4d_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py
readonly v4e_burned_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py
readonly v4f_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py
readonly runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_role_directed_teacher_margin_v4g.py
readonly runtime_test_relative=methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_role_directed_teacher_margin_v4g.py
readonly release_manifest_relative=release-manifest-v4g.json
readonly expected_release_manifest_schema=v4g-vjepa2-role-directed-teacher-margin-detached-release-manifest-v1
readonly expected_release_manifest_status=V4G_DETACHED_RELEASE_MANIFEST_SEALED

readonly -a release_relative_files=(
  "${feature_authority_relative}"
  "${v2_runtime_relative}"
  "${v4a_runtime_relative}"
  "${extractor_relative}"
  "${v4c_runtime_relative}"
  "${v4d_runtime_relative}"
  "${v4e_burned_runtime_relative}"
  "${v4f_runtime_relative}"
  "${runtime_relative}"
  "${runtime_test_relative}"
  "${release_manifest_relative}"
)
readonly -a release_expected_shas=(
  "${expected_feature_authority_sha256}"
  "${expected_v2_runtime_sha256}"
  "${expected_v4a_runtime_sha256}"
  "${expected_extractor_sha256}"
  "${expected_v4c_runtime_sha256}"
  "${expected_v4d_runtime_sha256}"
  "${expected_v4e_burned_runtime_sha256}"
  "${expected_v4f_runtime_sha256}"
  "${expected_runtime_sha256}"
  "${expected_runtime_test_sha256}"
  "${expected_release_manifest_sha256}"
)

fail() {
  printf '[%s] ERROR: %s\n' "${tag}" "$*" >&2
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

if [[ "${1:-}" == --draft-audit ]]; then
  [[ $# -eq 1 ]] || fail "--draft-audit takes no additional arguments"
  printf '%s\n' \
    '{"controller":"v4g-vjepa2-role-directed-teacher-margin-exact5-parallel-v1","intentional_no_go":false,"launch_performed":false,"release_sealed":true,"python_pin_sealed":true,"controller_contract_complete":true,"runtime_pin_final":true,"runtime_test_pin_final":true,"release_manifest_sha256":"d1e0c42904057e14d47c87e746c32db375ba6ee6b006f813ea91cdb2daae4882","release_manifest_digest":"91adc268b8b86f083c36aa043e5c6f10956e720b03a3b5ebf1022b3775ac46a4","release_manifest_status":"V4G_DETACHED_RELEASE_MANIFEST_SEALED","release_tree_sha256":"e4e158e064ceb181345673c86a8fb275436ddd25edc74a3e7ef8f1c31d4f16ff","release_exact_files":11,"fixed_no_fallback":true,"preflight":{"normal_tests":36,"optimized_tests":36,"skips":0,"compile_opt_levels":[0,2],"runtime_and_test_ast_assert_count":0,"before_first_persistent_mkdir":true},"state_machine":["train-fold-exact5","verify-inner-barrier-exact1","evaluate-fold-exact5","aggregate-exact1","branch-seal-exact1"],"allfold_oof_exact0_on_any_train_or_barrier_failure":true,"official_controller_cli_caller_supplied_inner_barrier_or_fold_sha":false,"nfs_postseal_bounded_retry":true,"single_fd_o_nofollow_exact_tree_final_seal":true,"launch_or_remote_action_performed":false}'
  exit 0
fi

[[ "${release_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: v4-G release/runtime/tests/manifest contract is not sealed"
[[ "${python_pin_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: pinned AUH Python entity is not sealed"
[[ "${controller_contract_complete}" == true ]] || \
  fail "INTENTIONAL NO-GO: v4-G controller contract has not passed dual audit"
[[ "${expected_release_tree_sha256}" != "${placeholder_sha256}" \
    && "${expected_release_manifest_sha256}" != "${placeholder_sha256}" \
    && "${expected_runtime_sha256}" != "${placeholder_sha256}" \
    && "${expected_runtime_test_sha256}" != "${placeholder_sha256}" ]] || \
  fail "INTENTIONAL NO-GO: v4-G source/tree/manifest pins are placeholders"

require_plain_file() {
  local fn_path=$1 fn_expected_sha=$2 fn_expected_mode=$3 fn_label=$4
  [[ "${fn_path}" == /* && -f "${fn_path}" && ! -L "${fn_path}" ]] || \
    fail "${fn_label} is not an absolute plain file"
  [[ "$(stat -c '%a:%h' "${fn_path}")" == "${fn_expected_mode}:1" ]] || \
    fail "${fn_label} mode/link differs"
  [[ "$(sha256_file "${fn_path}")" == "${fn_expected_sha}" ]] || \
    fail "${fn_label} SHA differs"
}

require_release() {
  [[ $# -eq 2 ]] || fail "require_release argument count differs"
  local fn_root=$1 fn_python_bin=$2 fn_result
  if ! fn_result="$("${fn_python_bin}" -I -S -B - \
      "${fn_root}" "${expected_release_tree_sha256}" \
      "${expected_release_manifest_digest}" "${expected_release_manifest_schema}" \
      "${expected_release_manifest_status}" \
      "${release_relative_files[@]}" "${release_expected_shas[@]}" <<'PY'
from pathlib import Path
import hashlib, json, os, re, stat, sys

root = Path(sys.argv[1]); expected_tree = sys.argv[2]
expected_manifest_digest, manifest_schema, manifest_status = sys.argv[3:6]
relative_files = sys.argv[6:17]; expected_shas = sys.argv[17:28]
if (len(relative_files) != 11 or len(expected_shas) != 11
        or len(set(relative_files)) != 11 or len(set(expected_shas)) != 11
        or not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))):
    raise SystemExit("release root/argument closure differs")
root_info = root.lstat()
if not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode) != 0o555:
    raise SystemExit("release root mode differs")
expected_directories = {
    "methods", "methods/bernini_action_editing",
    "methods/bernini_action_editing/tests",
}
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def read_same_fd(path, expected_sha, label):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1 or before.st_size <= 0):
        raise SystemExit(label + " envelope differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    actual = digest.hexdigest()
    if (path.is_symlink() or actual != expected_sha
            or not (identity(before) == identity(opened) == identity(closed)
                    == identity(after))):
        raise SystemExit(label + " same-FD SHA/identity differs")
    return b"".join(chunks), before, actual

members = list(root.rglob("*"))
if any(path.is_symlink() or (not path.is_dir() and not path.is_file())
       for path in members):
    raise SystemExit("release contains symlink or special member")
actual_directories = {
    path.relative_to(root).as_posix() for path in members if path.is_dir()
}
actual_files = {
    path.relative_to(root).as_posix() for path in members if path.is_file()
}
if actual_directories != expected_directories or actual_files != set(relative_files):
    raise SystemExit("release exact11 membership differs")
if any(stat.S_IMODE((root / relative).lstat().st_mode) != 0o555
       for relative in expected_directories):
    raise SystemExit("release directory mode differs")
rows = []; raw_by_relative = {}
for relative, expected_sha in zip(relative_files, expected_shas):
    raw, info, actual = read_same_fd(root / relative, expected_sha, relative)
    raw_by_relative[relative] = raw
    rows.append({"path": relative, "sha256": actual, "size_bytes": info.st_size})
rows.sort(key=lambda row: row["path"])
tree = hashlib.sha256(json.dumps(
    rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
).encode("ascii")).hexdigest()
if tree != expected_tree:
    raise SystemExit("release exact11 tree digest differs")

def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError("duplicate manifest key")
        result[key] = value
    return result
def nonfinite(value):
    raise ValueError("nonfinite manifest constant: " + value)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
manifest_relative = "release-manifest-v4g.json"
manifest = json.loads(raw_by_relative[manifest_relative],
                      object_pairs_hook=pairs, parse_constant=nonfinite)
unsigned = dict(manifest) if type(manifest) is dict else {}
manifest_digest = unsigned.pop("manifest_digest", None)
payload = manifest.get("payload") if type(manifest) is dict else None
expected_payload = [
    {"relative_path": relative_files[index], "sha256": expected_shas[index]}
    for index in range(10)
]
if (manifest.get("schema_version") != manifest_schema
        or manifest.get("status") != manifest_status
        or manifest_digest != expected_manifest_digest
        or object_sha(unsigned) != manifest_digest
        or manifest.get("payload_count") != 10 or type(payload) is not list
        or [{"relative_path": row.get("relative_path"),
             "sha256": row.get("sha256")} for row in payload] != expected_payload
        or manifest.get("manifest_target_relative_path") != manifest_relative
        or manifest.get("release_tree_contract", {}).get(
            "exact_file_count_including_manifest") != 11
        or manifest.get("authority_graph", {}).get(
            "sha256_graph_is_directed_and_acyclic") is not True
        or manifest.get("authority_graph", {}).get(
            "runtime_pins_controller_or_manifest") is not False):
    raise SystemExit("release manifest semantic closure differs")
print("V4G_RELEASE_EXACT11_VALIDATED=" + tree)
PY
    )"; then
    fail "release exact11/manifest validator failed"
  fi
  [[ "${fn_result}" == "V4G_RELEASE_EXACT11_VALIDATED=${expected_release_tree_sha256}" ]] || \
    fail "release exact11 validator output differs"
}

input_snapshot() {
  [[ $# -eq 6 ]] || fail "input_snapshot argument count differs"
  local fn_python_bin=$1 fn_feature_root=$2 fn_v4a_receipt=$3
  local fn_v4c_receipt=$4 fn_v4d_receipt=$5 fn_release_root=$6
  "${fn_python_bin}" -I -S -B - \
    "${fn_feature_root}" "${expected_feature_receipt_sha256}" \
    "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" \
    "${fn_v4c_receipt}" "${expected_v4c_frontier_receipt_sha256}" \
    "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" \
    "${fn_release_root}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

feature_root = Path(sys.argv[1]); release = Path(sys.argv[9])
authorities = [
    (feature_root / "feature_extraction_receipt.json", sys.argv[2], "feature"),
    (Path(sys.argv[3]), sys.argv[4], "v4a"),
    (Path(sys.argv[5]), sys.argv[6], "v4c"),
    (Path(sys.argv[7]), sys.argv[8], "v4d"),
]
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
rows = []
def bind(path, expected, label, parse=False):
    if (not path.is_absolute() or path.is_symlink()
            or str(path) != str(path.resolve(strict=True))):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1):
        raise SystemExit(label + " seal differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat(); raw = b"".join(chunks)
    if (not (identity(before) == identity(opened) == identity(closed)
             == identity(after)) or digest.hexdigest() != expected):
        raise SystemExit(label + " same-FD binding differs")
    rows.append({"label": label, "path": str(path), "sha256": expected,
                 "size_bytes": before.st_size})
    return json.loads(raw) if parse else None

feature = None
for path, expected, label in authorities:
    value = bind(path, expected, label, parse=label == "feature")
    if label == "feature": feature = value
if (not isinstance(feature, dict) or not isinstance(feature.get("shards"), list)
        or len(feature["shards"]) != 6):
    raise SystemExit("feature exact6 receipt differs")
for index, shard in enumerate(feature["shards"]):
    if not isinstance(shard, dict) or shard.get("index") != index:
        raise SystemExit("feature shard order differs")
    bind(Path(shard["path"]), shard["sha256"], "feature-shard-" + str(index))
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"),
                 ensure_ascii=True).encode("ascii")
print(hashlib.sha256(raw).hexdigest())
PY
}

snapshot_authorities() {
  [[ $# -eq 8 ]] || fail "snapshot_authorities argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_receipt=$5 fn_v4d_receipt=$6 fn_controller=$7 fn_controller_sha=$8
  local fn_input_digest
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python
  require_release "${fn_release_root}" "${fn_python_bin}"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 detached-controller
  require_plain_file "${fn_feature_root}/feature_extraction_receipt.json" \
    "${expected_feature_receipt_sha256}" 444 feature-receipt
  require_plain_file "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" 444 v4a-receipt
  require_plain_file "${fn_v4c_receipt}" "${expected_v4c_frontier_receipt_sha256}" 444 v4c-receipt
  require_plain_file "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" 444 v4d-receipt
  fn_input_digest="$(input_snapshot "${fn_python_bin}" "${fn_feature_root}" \
    "${fn_v4a_receipt}" "${fn_v4c_receipt}" "${fn_v4d_receipt}" \
    "${fn_release_root}")"
  [[ "${fn_input_digest}" =~ ^[0-9a-f]{64}$ ]] || fail "input snapshot differs"
  printf '%s:%s:%s:%s\n' "${expected_release_tree_sha256}" "${fn_input_digest}" \
    "${fn_controller_sha}" "${expected_python_sha256}"
}

fresh_cache() {
  local fn_prefix=$1 fn_cache
  fn_cache="$(mktemp -d "/tmp/${fn_prefix}.XXXXXX")"
  [[ "${fn_cache}" == /tmp/* && -d "${fn_cache}" && ! -L "${fn_cache}" ]] || \
    fail "fresh cache creation differs"
  printf '%s\n' "${fn_cache}"
}

activate_cache() {
  local fn_cache=$1
  export TMPDIR="${fn_cache}/tmp" TMP="${fn_cache}/tmp" TEMP="${fn_cache}/tmp"
  export XDG_CACHE_HOME="${fn_cache}/xdg" HF_HOME="${fn_cache}/hf"
  export PYTHONPYCACHEPREFIX="${fn_cache}/pycache" TORCH_HOME="${fn_cache}/torch"
  export MIOPEN_USER_DB_PATH="${fn_cache}/user-db" MIOPEN_CUSTOM_CACHE_DIR="${fn_cache}/custom"
  export TRITON_CACHE_DIR="${fn_cache}/triton" TORCHINDUCTOR_CACHE_DIR="${fn_cache}/inductor"
  mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${PYTHONPYCACHEPREFIX}" \
    "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" "${MIOPEN_CUSTOM_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  chmod 0700 "${fn_cache}" "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" \
    "${PYTHONPYCACHEPREFIX}" "${TORCH_HOME}" "${MIOPEN_USER_DB_PATH}" \
    "${MIOPEN_CUSTOM_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
}

cleanup_cache() {
  local fn_cache=$1
  [[ "${fn_cache}" =~ ^/tmp/v4g-[A-Za-z0-9-]+\.[A-Za-z0-9]{6}$ \
      && -d "${fn_cache}" && ! -L "${fn_cache}" ]] || \
    fail "unsafe cache cleanup target"
  chmod -R u+w -- "${fn_cache}"
  rm -rf -- "${fn_cache}"
  [[ ! -e "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup failed"
}

gpu_gate() {
  [[ $# -eq 5 ]] || fail "gpu_gate argument count differs"
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4 fn_role=$5
  "${fn_python_bin}" -P -B - "${fn_job}" "${fn_node}" "${fn_cache}" "${fn_role}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, subprocess, sys, time, uuid

job, node, cache_text, role = sys.argv[1:]
cache = Path(cache_text)
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True)) or Path("/tmp") not in cache.parents):
    raise SystemExit("GPU fresh cache root differs")
expected_dirs = {"tmp", "xdg", "hf", "pycache", "torch", "user-db",
                 "custom", "triton", "inductor"}
paths = list(cache.rglob("*"))
if ({path.relative_to(cache).as_posix() for path in paths} != expected_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700 for path in paths)):
    raise SystemExit("GPU fresh cache exact9 differs")
expected_env = {
    "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp",
    "XDG_CACHE_HOME": cache / "xdg", "HF_HOME": cache / "hf",
    "PYTHONPYCACHEPREFIX": cache / "pycache", "TORCH_HOME": cache / "torch",
    "MIOPEN_USER_DB_PATH": cache / "user-db",
    "MIOPEN_CUSTOM_CACHE_DIR": cache / "custom",
    "TRITON_CACHE_DIR": cache / "triton",
    "TORCHINDUCTOR_CACHE_DIR": cache / "inductor",
}
if any(os.environ.get(key) != str(value) for key, value in expected_env.items()):
    raise SystemExit("GPU cache environment differs")
import torch
expected_slurm = {"SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
                  "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
                  "SLURM_CPUS_PER_TASK": "8"}
if any(os.environ.get(key) != value for key, value in expected_slurm.items()):
    raise SystemExit("GPU exact1 Slurm gate differs")
actual_node = socket.gethostname().split(".", 1)[0]
step = os.environ.get("SLURM_STEP_ID", "")
physical = os.environ.get("SLURM_STEP_GPUS", "")
if (actual_node != node or os.environ.get("SLURM_GPUS_ON_NODE") != "1"
        or re.fullmatch(r"[0-9]+", step) is None
        or re.fullmatch(r"[0-9]+", physical) is None):
    raise SystemExit("GPU node/step/token gate differs")
if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("Torch/ROCm runtime differs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("logical exact1 ROCm gate differs")
properties = torch.cuda.get_device_properties(0)
name = torch.cuda.get_device_name(0); logical_uuid = str(getattr(properties, "uuid", ""))
if name != "AMD Instinct MI210" or not logical_uuid:
    raise SystemExit("logical MI210 identity differs")
try:
    decoded = uuid.UUID(logical_uuid).bytes.decode("ascii").lower()
except (ValueError, UnicodeDecodeError) as error:
    raise SystemExit("logical UUID decode differs") from error
if re.fullmatch(r"[0-9a-f]{16}", decoded) is None:
    raise SystemExit("logical HSA unique ID differs")
probe_env = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONPYCACHEPREFIX"):
    probe_env.pop(key, None)
inventory = None; last = ""
for _ in range(3):
    probe = subprocess.run(["rocm-smi", "--showuniqueid", "--json"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=probe_env)
    last = probe.stdout + probe.stderr
    if probe.returncode == 0:
        try: candidate = json.loads(probe.stdout)
        except json.JSONDecodeError: candidate = None
        if isinstance(candidate, dict) and len(candidate) == 1:
            inventory = candidate; break
    time.sleep(0.5)
if inventory is None:
    raise SystemExit("physical UUID probe failed: " + last)
card, row = next(iter(inventory.items()))
unique = [str(value) for key, value in row.items() if "unique" in key.lower()]
normalize = lambda value: "".join(character for character in
    value.lower().replace("gpu-", "").replace("0x", "")
    if character in "0123456789abcdef")
if len(unique) != 1 or normalize(unique[0]) != decoded:
    raise SystemExit("logical/physical MI210 UUID join differs")
print("V4G_GPU_GATE=" + json.dumps({
    "schema_version": "v4g-vjepa2-exact1-gpu-gate-v1", "role": role,
    "job_id": int(job), "step_id": step, "node": actual_node,
    "ntasks": 1, "nnodes": 1, "cpus_per_task": 8,
    "logical_device_count": 1, "visible_gpu_count": 1,
    "slurm_step_gpu_token": int(physical), "device_name": name,
    "logical_uuid": logical_uuid, "decoded_logical_unique_id": decoded,
    "rocm_inventory_card": card, "physical_uuid": unique[0],
    "exact_one_uuid_join": True, "torch": str(torch.__version__),
    "torch_hip": str(torch.version.hip), "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
}

cpu_gate() {
  [[ $# -eq 7 ]] || fail "cpu_gate argument count differs"
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4 fn_role=$5 fn_schema=$6 fn_cpus=$7
  "${fn_python_bin}" -I -S -B - "${fn_job}" "${fn_node}" "${fn_cache}" \
    "${fn_role}" "${fn_schema}" "${fn_cpus}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, sys
job, node, cache_text, role, schema, cpus = sys.argv[1:]
cache = Path(cache_text)
expected_dirs = {"tmp", "xdg", "hf", "pycache", "torch", "user-db",
                 "custom", "triton", "inductor"}
paths = list(cache.rglob("*"))
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True)) or Path("/tmp") not in cache.parents
        or {path.relative_to(cache).as_posix() for path in paths} != expected_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700 for path in paths)):
    raise SystemExit("CPU fresh cache exact9 differs")
expected_slurm = {"SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
                  "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
                  "SLURM_CPUS_PER_TASK": cpus}
if any(os.environ.get(key) != value for key, value in expected_slurm.items()):
    raise SystemExit("CPU exact1 Slurm gate differs")
step = os.environ.get("SLURM_STEP_ID", "")
if (socket.gethostname().split(".", 1)[0] != node
        or re.fullmatch(r"[0-9]+", step) is None
        or os.environ.get("SLURM_GPUS_ON_NODE") not in (None, "", "0")
        or os.environ.get("SLURM_STEP_GPUS") not in (None, "")
        or any(os.environ.get(key) is not None for key in (
            "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL"))):
    raise SystemExit("CPU node/GPU-absence gate differs")
print("V4G_CPU_GATE=" + json.dumps({
    "schema_version": schema, "role": role, "job_id": int(job),
    "step_id": step, "node": node, "ntasks": 1, "nnodes": 1,
    "cpus_per_task": int(cpus), "gres": "none",
    "gpu_visibility_variables_absent": True, "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
}, sort_keys=True, separators=(",", ":")), flush=True)
PY
}

recheck_holders() {
  local fn_phase=$1 fn_index fn_projection
  for fn_index in 0 1 2 3 4; do
    fn_projection="$(squeue -h -j "${worker_jobs[${fn_index}]}" \
      -w "${worker_nodes[${fn_index}]}" -o '%T|%u' | tr -d ' ')"
    [[ "${fn_projection}" == RUNNING\|guangyi.chen ]] || \
      fail "${fn_phase}: holder fold${fn_index} differs"
  done
  fn_projection="$(squeue -h -j "${cpu_job}" -w "${cpu_node}" -o '%T|%u' | tr -d ' ')"
  [[ "${fn_projection}" == RUNNING\|guangyi.chen ]] || \
    fail "${fn_phase}: CPU holder differs"
}

nfs_wait_path() {
  [[ $# -eq 3 ]] || fail "nfs_wait_path argument count differs"
  local fn_path=$1 fn_kind=$2 fn_label=$3 fn_attempt fn_stat
  for fn_attempt in $(seq 1 20); do
    if [[ "${fn_kind}" == file && -f "${fn_path}" && ! -L "${fn_path}" ]]; then
      fn_stat="$(stat -c '%a:%h:%s' "${fn_path}" 2>/dev/null || true)"
      if [[ "${fn_stat}" =~ ^444:1:[1-9][0-9]*$ || "${fn_stat}" == 444:1:0 ]]; then
        printf 'V4G_NFS_POSTSEAL_VISIBLE label=%s attempt=%s stat=%s\n' \
          "${fn_label}" "${fn_attempt}" "${fn_stat}" >&2
        return 0
      fi
    elif [[ "${fn_kind}" == directory && -d "${fn_path}" && ! -L "${fn_path}" \
        && "$(stat -c %a "${fn_path}" 2>/dev/null || true)" == 555 ]]; then
      printf 'V4G_NFS_POSTSEAL_VISIBLE label=%s attempt=%s stat=555\n' \
        "${fn_label}" "${fn_attempt}" >&2
      return 0
    fi
    printf 'V4G_NFS_POSTSEAL_RETRY label=%s attempt=%s exists=%s stat=%s\n' \
      "${fn_label}" "${fn_attempt}" "$([[ -e "${fn_path}" ]] && printf true || printf false)" \
      "$(stat -c '%F:%a:%h:%s' "${fn_path}" 2>/dev/null || printf absent)" >&2
    sleep 1
  done
  fail "${fn_label} did not become postseal-visible after bounded retry"
}

consumer_wait_file() {
  [[ $# -eq 5 ]] || fail "consumer_wait_file argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_sha=$3 fn_size=$4 fn_label=$5
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_sha}" "${fn_size}" \
    "${fn_label}" <<'PY'
from pathlib import Path
import hashlib, os, stat, sys, time
path = Path(sys.argv[1]); expected_sha = sys.argv[2]
expected_size = int(sys.argv[3]); label = sys.argv[4]
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
last = "not attempted"
for attempt in range(1, 21):
    try:
        if (not path.is_absolute() or path.is_symlink()
                or str(path) != str(path.resolve(strict=True))
                or not hasattr(os, "O_NOFOLLOW")):
            raise RuntimeError("path/canonical/O_NOFOLLOW differs")
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode)
                or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1 or before.st_size != expected_size):
            raise RuntimeError("mode/nlink/size differs: " + repr(
                (oct(stat.S_IMODE(before.st_mode)), before.st_nlink, before.st_size)))
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(fd); digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk: break
                digest.update(chunk)
            closed = os.fstat(fd)
        finally:
            os.close(fd)
        after = path.lstat()
        if (not (identity(before) == identity(opened) == identity(closed)
                 == identity(after)) or digest.hexdigest() != expected_sha):
            raise RuntimeError("same-FD identity/SHA differs")
        print("V4G_CONSUMER_NFS_VISIBLE label=" + label + " attempt=" + str(attempt)
              + " mode=0444 nlink=1 size=" + str(expected_size)
              + " sha256=" + expected_sha, file=sys.stderr, flush=True)
        break
    except (FileNotFoundError, OSError, RuntimeError) as error:
        last = type(error).__name__ + ":" + str(error)
        print("V4G_CONSUMER_NFS_RETRY label=" + label + " attempt=" + str(attempt)
              + " diagnostic=" + last, file=sys.stderr, flush=True)
        if attempt == 20:
            raise SystemExit("consumer NFS timeout for " + label + ": " + last)
        time.sleep(1.0)
PY
}

consumer_wait_directory() {
  [[ $# -eq 4 ]] || fail "consumer_wait_directory argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_names_csv=$3 fn_label=$4
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_names_csv}" "${fn_label}" <<'PY'
from pathlib import Path
import os, stat, sys, time
path = Path(sys.argv[1]); expected = set(filter(None, sys.argv[2].split(",")))
label = sys.argv[3]; last = "not attempted"
for attempt in range(1, 21):
    try:
        if (not path.is_absolute() or path.is_symlink()
                or str(path) != str(path.resolve(strict=True))):
            raise RuntimeError("directory path/canonical differs")
        info = path.lstat()
        members = list(path.iterdir())
        if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o555
                or {member.name for member in members} != expected
                or any(member.is_symlink() or not member.is_file() for member in members)):
            raise RuntimeError("directory mode/exact membership differs")
        print("V4G_CONSUMER_NFS_VISIBLE label=" + label + " attempt=" + str(attempt)
              + " mode=0555 exact_members=" + str(len(expected)),
              file=sys.stderr, flush=True)
        break
    except (FileNotFoundError, OSError, RuntimeError) as error:
        last = type(error).__name__ + ":" + str(error)
        print("V4G_CONSUMER_NFS_RETRY label=" + label + " attempt=" + str(attempt)
              + " diagnostic=" + last, file=sys.stderr, flush=True)
        if attempt == 20:
            raise SystemExit("consumer directory NFS timeout for " + label + ": " + last)
        time.sleep(1.0)
PY
}

consumer_wait_parent_directory() {
  [[ $# -eq 5 ]] || fail "consumer_wait_parent_directory argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_mode=$3 fn_empty=$4 fn_label=$5
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_mode}" "${fn_empty}" \
    "${fn_label}" <<'PY'
from pathlib import Path
import stat, sys, time
path = Path(sys.argv[1]); expected_mode = int(sys.argv[2], 8)
empty_text, label = sys.argv[3:5]
if empty_text not in {"true", "false"}:
    raise SystemExit("consumer parent-directory empty selector differs")
require_empty = empty_text == "true"; last = "not attempted"
for attempt in range(1, 21):
    try:
        if (not path.is_absolute() or path.is_symlink()
                or str(path) != str(path.resolve(strict=True))):
            raise RuntimeError("canonical/non-symlink path differs")
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) != expected_mode):
            raise RuntimeError("directory type/mode differs")
        members = list(path.iterdir())
        if require_empty and members:
            raise RuntimeError("directory is not empty")
        print("V4G_CONSUMER_NFS_VISIBLE label=" + label + " attempt="
              + str(attempt) + " mode=" + format(expected_mode, "04o")
              + " empty=" + str(not members).lower(), file=sys.stderr, flush=True)
        break
    except (FileNotFoundError, OSError, RuntimeError) as error:
        last = type(error).__name__ + ":" + str(error)
        print("V4G_CONSUMER_NFS_RETRY label=" + label + " attempt="
              + str(attempt) + " diagnostic=" + last,
              file=sys.stderr, flush=True)
        if attempt == 20:
            raise SystemExit("consumer parent-directory NFS timeout for "
                             + label + ": " + last)
        time.sleep(1.0)
PY
}

consumer_wait_phase_tree() {
  [[ $# -eq 4 ]] || fail "consumer_wait_phase_tree argument count differs"
  local fn_python_bin=$1 fn_phase_path=$2 fn_phase_sha=$3 fn_phase_size=$4
  "${fn_python_bin}" -I -S -B - "${fn_phase_path}" "${fn_phase_sha}" \
    "${fn_phase_size}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys, time
phase_path = Path(sys.argv[1]); expected_phase_sha = sys.argv[2]
expected_phase_size = int(sys.argv[3]); root = phase_path.parent
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def read_bound(path, expected_sha, expected_size):
    if (not path.is_absolute() or path.is_symlink()
            or str(path) != str(path.resolve(strict=True))
            or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("bound path differs: " + str(path))
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1 or before.st_size != expected_size):
        raise RuntimeError("bound envelope differs: " + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if (not (identity(before) == identity(opened) == identity(closed) == identity(after))
            or digest.hexdigest() != expected_sha):
        raise RuntimeError("bound same-FD identity/SHA differs: " + str(path))
    return b"".join(chunks)
last = "not attempted"
for attempt in range(1, 21):
    try:
        phase_raw = read_bound(phase_path, expected_phase_sha, expected_phase_size)
        phase = json.loads(phase_raw)
        rows = phase.get("pre_phase_state_files")
        directories = phase.get("expected_directories")
        if type(rows) is not list or type(directories) is not list:
            raise RuntimeError("phase exact-tree authority differs")
        expected_files = {row.get("path") for row in rows} | {"phase-state.json"}
        members = list(root.rglob("*"))
        actual_files = {path.relative_to(root).as_posix() for path in members if path.is_file()}
        actual_directories = {path.relative_to(root).as_posix()
                              for path in members if path.is_dir()}
        if (any(path.is_symlink() or (not path.is_dir() and not path.is_file())
                for path in members) or actual_files != expected_files
                or actual_directories != set(directories)
                or stat.S_IMODE(root.lstat().st_mode) != 0o555
                or any(stat.S_IMODE((root / relative).lstat().st_mode) != 0o555
                       for relative in directories)):
            raise RuntimeError("phase tree exact membership differs")
        for row in rows:
            read_bound(root / row["path"], row["sha256"], int(row["size_bytes"]))
        print("V4G_CONSUMER_NFS_VISIBLE label=seal-phase-tree attempt="
              + str(attempt) + " exact_files=" + str(len(expected_files))
              + " exact_directories=" + str(len(directories)),
              file=sys.stderr, flush=True)
        break
    except (FileNotFoundError, OSError, RuntimeError, ValueError,
            json.JSONDecodeError) as error:
        last = type(error).__name__ + ":" + str(error)
        print("V4G_CONSUMER_NFS_RETRY label=seal-phase-tree attempt=" + str(attempt)
              + " diagnostic=" + last, file=sys.stderr, flush=True)
        if attempt == 20:
            raise SystemExit("consumer phase/tree NFS timeout: " + last)
        time.sleep(1.0)
PY
}

seal_log_pair() {
  local fn_stdout=$1 fn_stderr=$2 fn_label=$3
  [[ -f "${fn_stdout}" && ! -L "${fn_stdout}" \
      && -f "${fn_stderr}" && ! -L "${fn_stderr}" ]] || \
    fail "${fn_label} log pair differs"
  chmod 0444 "${fn_stdout}" "${fn_stderr}"
  nfs_wait_path "${fn_stdout}" file "${fn_label}-stdout"
  nfs_wait_path "${fn_stderr}" file "${fn_label}-stderr"
}
verify_train_fold() {
  [[ $# -eq 4 ]] || fail "verify_train_fold argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_fold_root=$3 fn_fold=$4
  "${fn_python_bin}" -I -S -B - \
    "${fn_fold_root}" "${fn_fold}" "${expected_inner_schema}" \
    "${expected_inner_pass_status}" "${expected_inner_no_go_status}" \
    "${expected_checkpoint_schema}" \
    "${fn_release_root}/${runtime_relative}" "${expected_runtime_sha256}" \
    "${fn_release_root}/${v4f_runtime_relative}" "${expected_v4f_runtime_sha256}" \
    "${expected_v4c_runtime_sha256}" "${expected_extractor_sha256}" \
    "${expected_v4a_runtime_sha256}" "${expected_v4d_runtime_sha256}" \
    "${expected_v4e_burned_runtime_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, os, re, stat, sys

root = Path(sys.argv[1]); fold_index = int(sys.argv[2])
inner_schema, pass_status, no_go_status, checkpoint_schema = sys.argv[3:7]
implementation = {
    "implementation_path": str(Path(sys.argv[7]).resolve(strict=True)),
    "implementation_sha256": sys.argv[8],
    "frozen_v4f_runtime_path": str(Path(sys.argv[9]).resolve(strict=True)),
    "frozen_v4f_runtime_sha256": sys.argv[10],
    "v4c_implementation_sha256": sys.argv[11],
    "extractor_implementation_sha256": sys.argv[12],
    "v4a_implementation_sha256": sys.argv[13],
    "v4d_implementation_sha256": sys.argv[14],
    "v4e_burned_implementation_sha256": sys.argv[15],
}
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or stat.S_IMODE(root.lstat().st_mode) != 0o555
        or {path.name for path in root.iterdir()}
           != {"preselection.pt", "fixed1200.pt", "inner.json"}):
    raise SystemExit("train fold exact3 sealed closure differs")
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def read_sealed(path, label):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1 or before.st_size <= 0):
        raise SystemExit(label + " envelope differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " same-FD identity differs")
    return b"".join(chunks), before, digest.hexdigest()
def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result
def nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)
def object_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()

raw, receipt_info, receipt_sha = read_sealed(root / "inner.json", "inner receipt")
receipt = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
unsigned = dict(receipt) if type(receipt) is dict else {}
receipt_digest = unsigned.pop("receipt_digest", None)
pre_raw, pre_info, pre_sha = read_sealed(root / "preselection.pt", "preselection")
fixed_raw, fixed_info, fixed_sha = read_sealed(root / "fixed1200.pt", "fixed1200")
pre = receipt.get("preselection_checkpoint_artifact") if type(receipt) is dict else None
fixed = receipt.get("fixed1200_checkpoint_artifact") if type(receipt) is dict else None
pair = receipt.get("preselection_fixed1200_checkpoint_pair_join") if type(receipt) is dict else None
training = receipt.get("training") if type(receipt) is dict else None
candidate = receipt.get("fixed_candidate") if type(receipt) is dict else None
candidate_evidence = candidate.get("inner_evidence") if type(candidate) is dict else None
candidate_gate = candidate.get("gate") if type(candidate) is dict else None
bootstrap_ledger = candidate.get("bootstrap_seed_ledger") if type(candidate) is dict else None
fit_iids = receipt.get("model_fit_ordered_iids") if type(receipt) is dict else None
inner_iids = receipt.get("inner_validation_ordered_iids") if type(receipt) is dict else None
oof_iids = receipt.get("oof_ordered_iids") if type(receipt) is dict else None
inner_pass = receipt.get("inner_pass") if type(receipt) is dict else None
expected_status = pass_status if inner_pass is True else no_go_status if inner_pass is False else None
oof_counts = [131, 127, 128, 129, 129]
config = receipt.get("config") if type(receipt) is dict else None
materialization = (receipt.get("selective_feature_materialization_before_global_barrier")
                   if type(receipt) is dict else None)

def artifact_ok(value, path, info, sha, role):
    physical = {"device": info.st_dev, "inode": info.st_ino,
                "size_bytes": info.st_size}
    return (type(value) is dict and value.get("path") == str(path.resolve(strict=True))
            and value.get("file_sha256") == sha
            and value.get("size_bytes") == info.st_size
            and value.get("mode_octal") == "0444" and value.get("nlink") == 1
            and value.get("physical_identity") == physical
            and value.get("outer_fold") == fold_index
            and value.get("checkpoint_role") == role
            and value.get("fixed_step") == 1200
            and value.get("fixed_residual_scale") == 1.0
            and value.get("implementation_sha256") == implementation["implementation_sha256"]
            and re.fullmatch(r"[0-9a-f]{64}", str(value.get("metadata_digest"))) is not None
            and re.fullmatch(r"[0-9a-f]{64}", str(value.get("model_state_sha256"))) is not None
            and value.get("single_fd_pre_post_sha256_exact") is True
            and value.get("semantic_metadata_state_replay_verified") is True
            and value.get("fresh_reload_strict_state_verified") is True
            and value.get("fresh_reload_output_bit_exact") is True
            and value.get("caller_model_reloaded_from_sealed_artifact_before_next_stage") is True)

if (type(receipt) is not dict or receipt.get("schema_version") != inner_schema
        or expected_status is None or receipt.get("status") != expected_status
        or receipt.get("implementation") != implementation
        or receipt_digest != object_sha(unsigned)
        or receipt.get("fold_index") != fold_index
        or receipt.get("candidate_count") != 1
        or receipt.get("single_candidate") is not True
        or receipt.get("hyperparameter_selection_performed") is not False
        or type(config) is not dict or config.get("fixed_step") != 1200
        or config.get("fixed_residual_scale") != 1.0
        or config.get("candidate_count") != 1 or config.get("single_candidate") is not True
        or config.get("teacher_margin_weight") != 0.25
        or config.get("teacher_margin_beta") != 0.1
        or type(training) is not dict or type(candidate) is not dict
        or training.get("full_budget_steps_executed") != 1200
        or training.get("early_stopped") is not False
        or training.get("fixed_step") != 1200
        or training.get("checkpoint_winner_selection_performed") is not False
        or training.get("hyperparameter_selection_performed") is not False
        or training.get("trainable_parameter_count") != 79040
        or training.get("exact_three_role_directed_decoded_teacher_margins") is not True
        or training.get("teacher_margin_scale_mean_all_ten_teacher_distances_plus_1e_minus_8") is not True
        or training.get("oof_tensors_supplied_to_optimizer_checkpoint_or_inner_gate") is not False
        or type(fit_iids) is not list or type(inner_iids) is not list
        or type(oof_iids) is not list
        or len(set(fit_iids)) != len(fit_iids)
        or len(set(inner_iids)) != len(inner_iids)
        or len(set(oof_iids)) != len(oof_iids)
        or len(set(fit_iids + inner_iids + oof_iids)) != 644
        or len(oof_iids) != oof_counts[fold_index]
        or object_sha(fit_iids) != receipt.get("model_fit_iid_digest")
        or object_sha(inner_iids) != receipt.get("inner_validation_iid_digest")
        or object_sha(oof_iids) != receipt.get("oof_iid_digest")
        or candidate.get("candidate_count") != 1
        or candidate.get("single_candidate") is not True
        or candidate.get("fixed_step") != 1200
        or candidate.get("fixed_residual_scale") != 1.0
        or candidate.get("pass") is not inner_pass
        or candidate.get("inner_pass") is not inner_pass
        or type(candidate_evidence) is not list or not candidate_evidence
        or candidate.get("inner_evidence_count") != len(candidate_evidence)
        or candidate.get("inner_evidence_sha256") != object_sha(candidate_evidence)
        or [row.get("iid") for row in candidate_evidence] != inner_iids
        or any(row.get("fixed_step") != 1200
               or row.get("fixed_residual_scale") != 1.0
               or row.get("single_fixed_candidate") is not True
               for row in candidate_evidence)
        or type(candidate_gate) is not dict
        or candidate_gate.get("complete_candidate_dependent_inner_gate") is not inner_pass
        or type(bootstrap_ledger) is not list or len(bootstrap_ledger) != 34
        or candidate.get("model_state_sha256_before_inner")
           != candidate.get("model_state_sha256_after_inner")
        or candidate.get("model_state_sha256_before_inner")
           != fixed.get("model_state_sha256")
        or candidate.get("model_state_unchanged_during_inner_evaluation") is not True
        or type(materialization) is not dict
        or materialization.get("stage1_model_fit_only", {}).get(
            "semantic_tensor_materialized_count") != 5 * len(fit_iids)
        or materialization.get("stage2_post_both_checkpoint_seals_inner_only", {}).get(
            "semantic_tensor_materialized_count") != 5 * len(inner_iids)
        or materialization.get("stage1_oof_semantic_tensor_count") != 0
        or materialization.get("stage2_oof_semantic_tensor_count") != 0
        or receipt.get("oof_used_for_training_checkpoint_or_inner_gate") is not False
        or receipt.get("oof_semantic_tensor_materialized_count") != 0
        or receipt.get("oof_semantic_tensor_read_count_exact0") is not True
        or receipt.get("global_barrier_required_before_any_fold_oof") is not True
        or not artifact_ok(pre, root / "preselection.pt", pre_info, pre_sha,
                           "preselection_fixed_step1200")
        or not artifact_ok(fixed, root / "fixed1200.pt", fixed_info, fixed_sha,
                           "fixed1200_candidate")
        or pre.get("model_state_sha256") != fixed.get("model_state_sha256")
        or fixed.get("preselection_checkpoint_file_sha256") != pre_sha
        or fixed.get("preselection_checkpoint_binding") != {
            key: pre[key] for key in ("path", "file_sha256", "size_bytes",
                                      "mode_octal", "nlink", "metadata_digest",
                                      "model_state_sha256", "physical_identity")}
        or fixed.get("preselection_checkpoint_binding_sha256")
           != object_sha(fixed.get("preselection_checkpoint_binding"))
        or type(pair) is not dict
        or pair.get("preselection_device_inode") != [pre_info.st_dev, pre_info.st_ino]
        or pair.get("fixed1200_device_inode") != [fixed_info.st_dev, fixed_info.st_ino]
        or pair.get("distinct_device_inode_pair") is not True
        or [pre_info.st_dev, pre_info.st_ino] == [fixed_info.st_dev, fixed_info.st_ino]
        or pair.get("same_model_state_sha256") is not True
        or receipt.get("qualification_scope", {}).get("inner_fold_local_gate_passed")
           is not inner_pass
        or receipt.get("qualification_scope", {}).get("aggregate_gate_evaluated") is not False
        or receipt.get("qualification_scope", {}).get("inference_authorized") is not False):
    raise SystemExit("train fold semantic closure differs")
print("V4G_TRAIN_VERIFY=" + json.dumps({
    "fold_index": fold_index, "inner_pass": inner_pass,
    "inner_status": expected_status, "inner_receipt_sha256": receipt_sha,
    "inner_receipt_digest": receipt_digest,
    "inner_receipt_size_bytes": receipt_info.st_size,
    "preselection_sha256": pre_sha, "preselection_size_bytes": pre_info.st_size,
    "fixed1200_sha256": fixed_sha, "fixed1200_size_bytes": fixed_info.st_size,
    "same_model_state_sha256": True, "distinct_checkpoint_inodes": True,
    "oof_semantic_tensor_read_count": 0,
}, sort_keys=True, separators=(",", ":")))
PY
}
verify_barrier() {
  [[ $# -eq 9 ]] || fail "verify_barrier argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_run_root=$3 fn_barrier=$4
  shift 4
  "${fn_python_bin}" -I -S -B - \
    "${fn_run_root}" "${fn_barrier}" "${expected_barrier_schema}" \
    "${expected_barrier_pass_status}" \
    "${fn_release_root}/${runtime_relative}" "${expected_runtime_sha256}" \
    "${fn_release_root}/${v4f_runtime_relative}" "${expected_v4f_runtime_sha256}" \
    "$@" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

run_root = Path(sys.argv[1]); path = Path(sys.argv[2])
schema, status = sys.argv[3:5]
implementation = {
    "implementation_path": str(Path(sys.argv[5]).resolve(strict=True)),
    "implementation_sha256": sys.argv[6],
    "frozen_v4f_runtime_path": str(Path(sys.argv[7]).resolve(strict=True)),
    "frozen_v4f_runtime_sha256": sys.argv[8],
    "v4c_implementation_sha256": "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef",
    "extractor_implementation_sha256": "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc",
    "v4a_implementation_sha256": "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973",
    "v4d_implementation_sha256": "20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc",
    "v4e_burned_implementation_sha256": "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a",
}
expected_inner_shas = sys.argv[9:14]
if (len(expected_inner_shas) != 5 or len(set(expected_inner_shas)) != 5
        or not run_root.is_absolute() or run_root.is_symlink()
        or str(run_root) != str(run_root.resolve(strict=True))
        or path != run_root / "barrier" / "barrier.json"):
    raise SystemExit("barrier arguments differ")
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def read_json(path, label):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1 or before.st_size <= 0):
        raise SystemExit(label + " envelope differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " same-FD identity differs")
    raw = b"".join(chunks)
    return json.loads(raw), before, digest.hexdigest()
def object_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()

value, info, file_sha = read_json(path, "barrier")
unsigned = dict(value) if type(value) is dict else {}
receipt_digest = unsigned.pop("receipt_digest", None)
members = value.get("members") if type(value) is dict else None
replay = value.get("independent_replay_ledger") if type(value) is dict else None
if (type(value) is not dict or value.get("schema_version") != schema
        or value.get("status") != status or value.get("implementation") != implementation
        or receipt_digest != object_sha(unsigned)
        or type(members) is not list or len(members) != 5
        or [row.get("fold_index") for row in members] != list(range(5))
        or [row.get("fold_root") for row in members]
           != [str((run_root / ("fold" + str(index))).resolve(strict=True))
               for index in range(5)]
        or [row.get("inner_receipt_binding", {}).get("file_sha256")
            for row in members] != expected_inner_shas
        or any(row.get("inner_pass") is not True
               or row.get("oof_semantic_tensor_read_count_exact0") is not True
               for row in members)
        or value.get("members_sha256") != object_sha(members)
        or type(replay) is not list or len(replay) != 5
        or [row.get("fold_index") for row in replay] != list(range(5))
        or value.get("independent_replay_sha256") != object_sha(replay)
        or any(row.get("inner_pass") is not True
               or row.get("oof_semantic_tensor_read_count") != 0
               or row.get("checkpoint_outer_fold_authority_join") is not True
               or row.get("checkpoint_model_fit_ordered_iids_authority_join") is not True
               or row.get("checkpoint_model_fit_count_and_digest_authority_join") is not True
               or row.get("checkpoint_inner_iid_digest_authority_join") is not True
               or row.get("checkpoint_pca_fit_input_receipt_training_join") is not True
               or row.get("checkpoint_minibatch_schedule_receipt_training_join") is not True
               or row.get("checkpoint_state_receipt_training_inner_join") is not True
               or row.get("checkpoint_clip_mean_equals_authority_recomputed_pca") is not True
               or row.get("checkpoint_clip_basis_equals_authority_recomputed_pca") is not True
               or row.get("checkpoint_fit_only_rms_equals_authority_recomputed_rms") is not True
               or row.get("checkpoint_schema_and_exact79040_strict_loaded") is not True
               or row.get("full_candidate_ledger_exact_match") is not True
               for row in replay)
        or value.get("all_five_exact_one_full_gates_pass") is not True
        or value.get("all_five_authority_model_fit_provenances_recomputed") is not True
        or value.get("all_five_authority_inner_checkpoint_forwards_reexecuted") is not True
        or value.get("oof_semantic_tensor_read_count") != 0
        or value.get("oof_semantic_tensor_read_count_exact0") is not True
        or value.get("evaluate_fold_accepts_only_this_barrier_path_and_controller_expected_sha") is not True
        or value.get("arbitrary_evaluate_fold_child_roots_or_inner_shas_accepted") is not False):
    raise SystemExit("barrier semantic closure differs")
print("V4G_BARRIER_VERIFY=" + json.dumps({
    "barrier_receipt": str(path.resolve(strict=True)),
    "barrier_receipt_sha256": file_sha, "barrier_receipt_digest": receipt_digest,
    "barrier_receipt_size_bytes": info.st_size,
    "inner_receipt_shas": expected_inner_shas,
    "model_fit_pca_rms_schedule_and_inner_replay_exact5": True,
    "oof_semantic_tensor_read_count": 0,
}, sort_keys=True, separators=(",", ":")))
PY
}
verify_evaluate_fold() {
  [[ $# -eq 8 ]] || fail "verify_evaluate_fold argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_fold_root=$3 fn_fold=$4
  local fn_barrier_path=$5 fn_barrier_sha=$6 fn_barrier_digest=$7 fn_inner_sha=$8
  "${fn_python_bin}" -I -S -B - \
    "${fn_fold_root}" "${fn_fold}" "${fn_barrier_path}" "${fn_barrier_sha}" \
    "${fn_barrier_digest}" "${fn_inner_sha}" "${expected_fold_schema}" \
    "${expected_aggregate_status}" \
    "${fn_release_root}/${runtime_relative}" "${expected_runtime_sha256}" \
    "${fn_release_root}/${v4f_runtime_relative}" "${expected_v4f_runtime_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

root = Path(sys.argv[1]); fold_index = int(sys.argv[2]); barrier_path = Path(sys.argv[3])
barrier_sha, barrier_digest, inner_sha = sys.argv[4:7]
schema, status = sys.argv[7:9]
implementation = {
    "implementation_path": str(Path(sys.argv[9]).resolve(strict=True)),
    "implementation_sha256": sys.argv[10],
    "frozen_v4f_runtime_path": str(Path(sys.argv[11]).resolve(strict=True)),
    "frozen_v4f_runtime_sha256": sys.argv[12],
    "v4c_implementation_sha256": "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef",
    "extractor_implementation_sha256": "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc",
    "v4a_implementation_sha256": "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973",
    "v4d_implementation_sha256": "20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc",
    "v4e_burned_implementation_sha256": "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a",
}
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or stat.S_IMODE(root.lstat().st_mode) != 0o555
        or {path.name for path in root.iterdir()}
           != {"preselection.pt", "fixed1200.pt", "inner.json", "fold.json"}):
    raise SystemExit("evaluated fold exact4 sealed closure differs")
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def read_json(path, label):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1 or before.st_size <= 0):
        raise SystemExit(label + " envelope differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " same-FD identity differs")
    return json.loads(b"".join(chunks)), before, digest.hexdigest()
def object_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()

value, info, file_sha = read_json(root / "fold.json", "fold receipt")
unsigned = dict(value) if type(value) is dict else {}
receipt_digest = unsigned.pop("receipt_digest", None)
barrier = value.get("global_inner_barrier") if type(value) is dict else None
evidence = value.get("oof_evidence") if type(value) is dict else None
fixed_artifact = value.get("fixed1200_checkpoint_artifact") if type(value) is dict else None
fixed_binding = value.get("fixed1200_evaluate_checkpoint_binding") if type(value) is dict else None
oof_counts = [131, 127, 128, 129, 129]
if (type(value) is not dict or value.get("schema_version") != schema
        or value.get("status") != status or value.get("implementation") != implementation
        or receipt_digest != object_sha(unsigned) or value.get("fold_index") != fold_index
        or value.get("inner_receipt_file_sha256") != inner_sha
        or value.get("controller_barrier_receipt_file_sha256") != barrier_sha
        or type(barrier) is not dict
        or barrier.get("controller_barrier_receipt_binding", {}).get("path")
           != str(barrier_path.resolve(strict=True))
        or barrier.get("controller_barrier_receipt_binding", {}).get("file_sha256")
           != barrier_sha
        or barrier.get("controller_barrier_receipt_digest") != barrier_digest
        or barrier.get("exact_five_inner_receipts_verified") is not True
        or barrier.get("all_five_exact_one_full_gates_pass") is not True
        or barrier.get("all_five_checkpoints_independently_forward_replayed") is not True
        or barrier.get("all_five_authority_inner_populations_re_materialized") is not True
        or barrier.get("all_five_full_candidate_ledgers_exact_match") is not True
        or barrier.get("barrier_completed_before_any_oof_tensor_request") is not True
        or type(barrier.get("independent_replay_ledger")) is not list
        or len(barrier["independent_replay_ledger"]) != 5
        or barrier.get("independent_replay_sha256")
           != object_sha(barrier["independent_replay_ledger"])
        or type(evidence) is not list or len(evidence) != oof_counts[fold_index]
        or value.get("oof_original_count") != oof_counts[fold_index]
        or value.get("oof_evidence_count") != oof_counts[fold_index]
        or value.get("oof_evidence_sha256") != object_sha(evidence)
        or [row.get("iid") for row in evidence] != value.get("oof_ordered_iids")
        or len({row.get("iid") for row in evidence}) != oof_counts[fold_index]
        or any(row.get("outer_fold") != fold_index
               or row.get("fixed_step") != 1200
               or row.get("fixed_residual_scale") != 1.0
               or row.get("single_fixed_candidate") is not True for row in evidence)
        or type(fixed_artifact) is not dict or type(fixed_binding) is not dict
        or fixed_binding.get("file_sha256") != fixed_artifact.get("file_sha256")
        or fixed_binding.get("model_state_sha256") != fixed_artifact.get("model_state_sha256")
        or fixed_binding.get("outer_fold") != fold_index
        or value.get("oof_used_for_training_checkpoint_inner_gate_or_selection") is not False
        or value.get("qualification_scope", {}).get("inference_authorized") is not False
        or value.get("qualification_scope", {}).get("aggregate_gate_evaluated") is not False):
    raise SystemExit("evaluated fold semantic closure differs")
print("V4G_EVALUATE_VERIFY=" + json.dumps({
    "fold_index": fold_index, "fold_receipt_sha256": file_sha,
    "fold_receipt_digest": receipt_digest, "fold_receipt_size_bytes": info.st_size,
    "barrier_receipt_sha256": barrier_sha, "inner_receipt_sha256": inner_sha,
    "oof_original_count": len(evidence), "post_barrier_oof_only": True,
}, sort_keys=True, separators=(",", ":")))
PY
}
verify_aggregate() {
  [[ $# -eq 10 ]] || fail "verify_aggregate argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_run_root=$3
  local fn_barrier_sha=$4 fn_barrier_digest=$5
  shift 5
  "${fn_python_bin}" -I -S -B - \
    "${fn_run_root}" "${fn_barrier_sha}" "${fn_barrier_digest}" \
    "${expected_aggregate_schema}" "${expected_aggregate_status}" \
    "${fn_release_root}/${runtime_relative}" "${expected_runtime_sha256}" \
    "${fn_release_root}/${v4f_runtime_relative}" "${expected_v4f_runtime_sha256}" \
    "$@" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

run_root = Path(sys.argv[1]); barrier_sha, barrier_digest = sys.argv[2:4]
schema, status = sys.argv[4:6]
implementation = {
    "implementation_path": str(Path(sys.argv[6]).resolve(strict=True)),
    "implementation_sha256": sys.argv[7],
    "frozen_v4f_runtime_path": str(Path(sys.argv[8]).resolve(strict=True)),
    "frozen_v4f_runtime_sha256": sys.argv[9],
    "v4c_implementation_sha256": "d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef",
    "extractor_implementation_sha256": "720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc",
    "v4a_implementation_sha256": "e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973",
    "v4d_implementation_sha256": "20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc",
    "v4e_burned_implementation_sha256": "4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a",
}
expected_fold_shas = sys.argv[10:15]
path = run_root / "aggregate" / "receipt.json"
if (len(expected_fold_shas) != 5 or len(set(expected_fold_shas)) != 5
        or not run_root.is_absolute() or run_root.is_symlink()
        or str(run_root) != str(run_root.resolve(strict=True))):
    raise SystemExit("aggregate arguments differ")
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
    raise SystemExit("aggregate receipt path differs")
before = path.lstat()
if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1 or before.st_size <= 0):
    raise SystemExit("aggregate receipt envelope differs")
fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk: break
        chunks.append(chunk); digest.update(chunk)
    closed = os.fstat(fd)
finally:
    os.close(fd)
after = path.lstat()
if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
    raise SystemExit("aggregate same-FD identity differs")
value = json.loads(b"".join(chunks))
object_sha = lambda item: hashlib.sha256(json.dumps(
    item, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).hexdigest()
unsigned = dict(value) if type(value) is dict else {}
receipt_digest = unsigned.pop("receipt_digest", None)
folds = value.get("folds") if type(value) is dict else None
fold_bindings = value.get("fold_receipts", {}).get("bindings") if type(value) is dict else None
oof = value.get("oof_closure") if type(value) is dict else None
evidence = oof.get("embedded_per_iid_evidence") if type(oof) is dict else None
metrics = value.get("metrics") if type(value) is dict else None
if (type(value) is not dict or value.get("schema_version") != schema
        or value.get("status") != status or value.get("implementation") != implementation
        or receipt_digest != object_sha(unsigned)
        or value.get("controller_barrier_receipt_binding", {}).get("file_sha256")
           != barrier_sha
        or value.get("controller_barrier_receipt_digest") != barrier_digest
        or type(folds) is not list or len(folds) != 5
        or [row.get("fold_index") for row in folds] != list(range(5))
        or type(fold_bindings) is not list or len(fold_bindings) != 5
        or [row.get("file_sha256") for row in fold_bindings] != expected_fold_shas
        or value.get("fold_receipts", {}).get("count") != 5
        or value.get("inner_receipts", {}).get("count") != 5
        or value.get("global_inner_barrier", {}).get(
            "exact_five_inner_receipts_verified") is not True
        or value.get("global_inner_barrier", {}).get(
            "all_five_exact_one_full_gates_pass") is not True
        or value.get("global_inner_barrier", {}).get(
            "every_train_fold_oof_read_count_exact0") is not True
        or type(oof) is not dict or type(evidence) is not list or len(evidence) != 644
        or len({row.get("iid") for row in evidence}) != 644
        or oof.get("unique_original_iids") != 644
        or oof.get("each_original_evaluated_exactly_once") is not True
        or oof.get("embedded_per_iid_evidence_count") != 644
        or oof.get("embedded_per_iid_evidence_sha256") != object_sha(evidence)
        or type(metrics) is not dict
        or type(metrics.get("exposed_five_view_codec_development_gate")) is not bool
        or value.get("qualification_scope", {}).get("aggregate_gate_evaluated") is not True
        or value.get("qualification_scope", {}).get("inference_authorized") is not False):
    raise SystemExit("aggregate semantic closure differs")
print("V4G_AGGREGATE_VERIFY=" + json.dumps({
    "aggregate_receipt": str(path.resolve(strict=True)),
    "aggregate_receipt_sha256": digest.hexdigest(),
    "aggregate_receipt_digest": receipt_digest,
    "aggregate_receipt_size_bytes": before.st_size,
    "barrier_receipt_sha256": barrier_sha,
    "fold_receipt_shas": expected_fold_shas,
    "oof_original_count": len(evidence),
    "exposed_five_view_codec_development_gate": metrics[
        "exposed_five_view_codec_development_gate"],
    "inference_authorized": False,
}, sort_keys=True, separators=(",", ":")))
PY
}
run_exact_test_suite() {
  [[ $# -eq 3 ]] || fail "run_exact_test_suite argument count differs"
  local fn_python_bin=$1 fn_optimized=$2 fn_test_module=$3
  local -a fn_flags=(-P -B)
  if [[ "${fn_optimized}" == true ]]; then
    fn_flags=(-O -P -B)
  elif [[ "${fn_optimized}" != false ]]; then
    fail "test optimization selector differs"
  fi
  "${fn_python_bin}" "${fn_flags[@]}" - "${fn_test_module}" <<'PY'
import importlib, json, sys, unittest
import torch
if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("preflight Torch/ROCm differs")
module = importlib.import_module(sys.argv[1])
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
summary = {
    "errors": len(result.errors), "expected_failures": len(result.expectedFailures),
    "failures": len(result.failures), "optimized": bool(sys.flags.optimize),
    "skipped": len(result.skipped), "tests_run": result.testsRun,
    "unexpected_successes": len(result.unexpectedSuccesses),
}
expected = {
    "errors": 0, "expected_failures": 0, "failures": 0,
    "optimized": bool(sys.flags.optimize), "skipped": 0, "tests_run": 36,
    "unexpected_successes": 0,
}
if summary != expected:
    raise SystemExit("preflight exact36 closure differs: " + repr(summary))
print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
PY
}

run_preflight_child() {
  [[ $# -eq 10 ]] || fail "preflight child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_receipt=$5 fn_v4d_receipt=$6 fn_job=$7 fn_node=$8
  local fn_controller_sha=$9 fn_snapshot=${10}
  local fn_controller fn_runtime fn_cache fn_now fn_gate fn_gate_json fn_test_module
  local fn_normal_out fn_normal_err fn_opt_out fn_opt_err fn_normal_pid fn_opt_pid
  local fn_normal_rc=0 fn_opt_rc=0 fn_help fn_token fn_heredoc_result
  fn_controller="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${preflight_job}" && "${fn_node}" == "${preflight_node}" \
      && "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "preflight holder binding differs"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 preflight-controller
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "preflight initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "v4g-preflight-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  fn_gate="$(cpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    preflight v4g-vjepa2-cpu-preflight-gate-v1 "${preflight_cpus}")"
  [[ "${fn_gate}" == V4G_CPU_GATE=* ]] || fail "preflight CPU gate output differs"
  fn_gate_json=${fn_gate#V4G_CPU_GATE=}
  fn_test_module=methods.bernini_action_editing.tests.test_semantic_anchor_vjepa2_role_directed_teacher_margin_v4g
  fn_normal_out="${fn_cache}/normal.stdout"; fn_normal_err="${fn_cache}/normal.stderr"
  fn_opt_out="${fn_cache}/optimized.stdout"; fn_opt_err="${fn_cache}/optimized.stderr"
  (
    set -o noclobber; cd "${fn_release_root}"
    run_exact_test_suite "${fn_python_bin}" false "${fn_test_module}" \
      >"${fn_normal_out}" 2>"${fn_normal_err}"
  ) &
  fn_normal_pid=$!
  (
    set -o noclobber; cd "${fn_release_root}"
    run_exact_test_suite "${fn_python_bin}" true "${fn_test_module}" \
      >"${fn_opt_out}" 2>"${fn_opt_err}"
  ) &
  fn_opt_pid=$!
  wait "${fn_normal_pid}" || fn_normal_rc=$?
  wait "${fn_opt_pid}" || fn_opt_rc=$?
  if (( fn_normal_rc != 0 || fn_opt_rc != 0 )); then
    sed -n '1,240p' "${fn_normal_out}" "${fn_normal_err}" \
      "${fn_opt_out}" "${fn_opt_err}" >&2
    fail "normal/-O exact36 preflight failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_normal_out}" "${fn_opt_out}" \
    "${fn_normal_err}" "${fn_opt_err}" <<'PY'
from pathlib import Path
import json, sys
common = {"errors": 0, "expected_failures": 0, "failures": 0,
          "skipped": 0, "tests_run": 36, "unexpected_successes": 0}
for index, optimized in enumerate((False, True), start=1):
    value = json.loads(Path(sys.argv[index]).read_text("ascii"))
    if value != {**common, "optimized": optimized}:
        raise SystemExit("test JSON closure differs")
for index in (3, 4):
    lines = Path(sys.argv[index]).read_text("utf-8").splitlines()
    if (not any(line.startswith("Ran 36 tests in ") for line in lines)
            or "OK" not in lines or any("skipped=" in line for line in lines)):
        raise SystemExit("unittest text closure differs")
PY
  (
    cd "${fn_release_root}"
    "${fn_python_bin}" -P -B -m py_compile \
      "${release_relative_files[@]:0:10}"
    "${fn_python_bin}" -OO -P -B -m py_compile \
      "${release_relative_files[@]:0:10}"
  )
  fn_heredoc_result="$("${fn_python_bin}" -I -S -B - \
    "${fn_release_root}/${runtime_relative}" \
    "${fn_release_root}/${runtime_test_relative}" "${fn_controller}" <<'PY'
from pathlib import Path
import ast, hashlib, json, re, sys
runtime, tests, controller = [Path(value) for value in sys.argv[1:4]]
sources = {"runtime": runtime.read_text("utf-8"),
           "tests": tests.read_text("utf-8")}
counts = {}
for name, source in sources.items():
    tree = ast.parse(source, filename=str(runtime if name == "runtime" else tests))
    counts[name] = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    compile(source, str(runtime if name == "runtime" else tests), "exec", optimize=0)
    compile(source, str(runtime if name == "runtime" else tests), "exec", optimize=2)
raw = controller.read_text("utf-8")
blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", raw, flags=re.DOTALL)
if not blocks:
    raise SystemExit("controller has no Python heredocs")
heredoc_asserts = 0
for index, block in enumerate(blocks):
    tree = ast.parse(block, filename=str(controller) + ":heredoc:" + str(index))
    heredoc_asserts += sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    compile(block, str(controller) + ":heredoc:" + str(index), "exec", optimize=0)
    compile(block, str(controller) + ":heredoc:" + str(index), "exec", optimize=2)
if counts != {"runtime": 0, "tests": 0} or heredoc_asserts != 0:
    raise SystemExit("AST assert closure differs")
stable = {"path": "fold0/fixed1200.pt", "sha256": "a" * 64,
          "size_bytes": 17, "mode_octal": "0444", "nlink": 1}
captured = {**stable, "physical_identity": {
    "device": 1, "inode": 11, "size_bytes": 17},
    "metadata_digest": "b" * 64, "model_state_sha256": "c" * 64}
replacement = {**captured, "physical_identity": {
    "device": 1, "inode": 12, "size_bytes": 17}}
barrier_members = [{"fold_index": index, "fixed1200_checkpoint_artifact":
                    captured if index == 0 else {**captured, "path":
                    "fold" + str(index) + "/fixed1200.pt"}}
                   for index in range(5)]
split_members = list(reversed(barrier_members))
launch_subset = {"schema_version": "v4g-exact5-role-directed-launch-plan-v1",
                 "controller_sha256": "d" * 64, "fixed_no_fallback": True}
launch_first = {**launch_subset, "cpu_preflight": {"normal_tests": 36},
                "workers": list(range(5)), "state_machine": ["train", "barrier"]}
launch_coherent_subset_replacement = {
    **launch_subset, "cpu_preflight": {"normal_tests": 0},
    "workers": [], "state_machine": ["aggregate"]}
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")
object_sha = lambda value: hashlib.sha256(canonical(value)).hexdigest()
launch_record = {"value": launch_first,
                 "file_binding": {"sha256": "e" * 64, "size_bytes": 211}}
launch_replacement_binding = {"sha256": "f" * 64, "size_bytes": 207}
def launch_record_matches(value, binding, record):
    return value == record["value"] and binding == record["file_binding"]
launch_replacement_rejected = not launch_record_matches(
    launch_coherent_subset_replacement, launch_replacement_binding, launch_record)
launch_original_accepted = launch_record_matches(
    launch_first, launch_record["file_binding"], launch_record)
aggregate_values = [{"role": "aggregate"}, {"gate": False},
                    {"aggregate_receipt_sha256": "1" * 64},
                    {"absent_after_cleanup": True}]
aggregate_unsigned = {
    "status": "V4G_ROLE_DIRECTED_TEACHER_MARGIN_KNOWN_EXPOSED_DEVELOPMENT",
    "metrics": {"exposed_five_view_codec_development_gate": False},
    "controller_barrier_receipt_binding": {"file_sha256": "2" * 64},
    "inner_receipts": {"count": 5, "bindings": ["3" * 64] * 5},
    "fold_receipts": {"count": 5, "bindings": ["4" * 64] * 5},
    "folds": [{"fold_index": index} for index in range(5)],
    "global_inner_barrier": {"file_sha256": "5" * 64},
}
aggregate_receipt = dict(aggregate_unsigned)
aggregate_receipt["receipt_digest"] = object_sha(aggregate_unsigned)
aggregate_files = [{"relative_path": "logs/aggregate.stdout", "sha256": "6" * 64},
                   {"relative_path": "aggregate/receipt.json", "sha256": "7" * 64}]
aggregate_record = {
    "files": aggregate_files,
    "structured_values_sha256": object_sha(aggregate_values),
    "receipt_digest": aggregate_receipt["receipt_digest"],
    "receipt_status": aggregate_receipt["status"],
    "exposed_five_view_codec_development_gate": False,
    "controller_barrier_receipt_binding_sha256": object_sha(
        aggregate_receipt["controller_barrier_receipt_binding"]),
    "inner_receipt_binding_ledger_sha256": object_sha(
        aggregate_receipt["inner_receipts"]),
    "fold_receipt_binding_ledger_sha256": object_sha(
        aggregate_receipt["fold_receipts"]),
    "fold_value_ledger_sha256": object_sha(aggregate_receipt["folds"]),
    "global_inner_barrier_sha256": object_sha(
        aggregate_receipt["global_inner_barrier"]),
}
def aggregate_record_matches(values, receipt, files, record):
    unsigned = dict(receipt); digest = unsigned.pop("receipt_digest", None)
    return (files == record["files"]
            and object_sha(values) == record["structured_values_sha256"]
            and digest == object_sha(unsigned) == record["receipt_digest"]
            and receipt.get("status") == record["receipt_status"]
            and receipt.get("metrics", {}).get(
                "exposed_five_view_codec_development_gate")
               is record["exposed_five_view_codec_development_gate"]
            and object_sha(receipt.get("controller_barrier_receipt_binding"))
               == record["controller_barrier_receipt_binding_sha256"]
            and object_sha(receipt.get("inner_receipts"))
               == record["inner_receipt_binding_ledger_sha256"]
            and object_sha(receipt.get("fold_receipts"))
               == record["fold_receipt_binding_ledger_sha256"]
            and object_sha(receipt.get("folds"))
               == record["fold_value_ledger_sha256"]
            and object_sha(receipt.get("global_inner_barrier"))
               == record["global_inner_barrier_sha256"])
replacement_values = [{"role": "aggregate"}, {"gate": True},
                      {"aggregate_receipt_sha256": "8" * 64},
                      {"absent_after_cleanup": True}]
replacement_unsigned = {
    **aggregate_unsigned,
    "metrics": {"exposed_five_view_codec_development_gate": True},
    "controller_barrier_receipt_binding": {"file_sha256": "9" * 64},
    "inner_receipts": {"count": 5, "bindings": ["a" * 64] * 5},
    "fold_receipts": {"count": 5, "bindings": ["b" * 64] * 5},
    "folds": [{"fold_index": index, "replacement": True} for index in range(5)],
    "global_inner_barrier": {"file_sha256": "c" * 64},
}
replacement_receipt = dict(replacement_unsigned)
replacement_receipt["receipt_digest"] = object_sha(replacement_unsigned)
replacement_files = [{"relative_path": "logs/aggregate.stdout", "sha256": "d" * 64},
                     {"relative_path": "aggregate/receipt.json", "sha256": "0" * 64}]
aggregate_replacement_rejected = not aggregate_record_matches(
    replacement_values, replacement_receipt, replacement_files, aggregate_record)
aggregate_original_accepted = aggregate_record_matches(
    aggregate_values, aggregate_receipt, aggregate_files, aggregate_record)
if (captured == replacement or barrier_members == split_members
        or {key: captured[key] for key in stable} != stable
        or {key: replacement[key] for key in stable} != stable
        or {key: launch_first[key] for key in launch_subset}
           != {key: launch_coherent_subset_replacement[key] for key in launch_subset}
        or not launch_original_accepted or not launch_replacement_rejected
        or not aggregate_original_accepted or not aggregate_replacement_rejected):
    raise SystemExit("hostile replacement/split binding selftest differs")
print(json.dumps({"runtime_ast_assert_count": 0, "tests_ast_assert_count": 0,
                  "controller_python_heredoc_count": len(blocks),
                  "controller_heredoc_ast_assert_count": 0,
                  "compile_opt_levels": [0, 2],
                  "same_node_byte_identical_new_inode_replacement_rejected": True,
                  "barrier_member_or_aggregate_ledger_split_rejected": True,
                  "launch_plan_subset_preserving_full_payload_replacement_rejected":
                      launch_replacement_rejected,
                  "launch_plan_original_history_join_accepted":
                      launch_original_accepted,
                  "aggregate_receipt_stdout_coherent_replacement_rejected":
                      aggregate_replacement_rejected,
                  "aggregate_original_history_join_accepted":
                      aggregate_original_accepted},
                 sort_keys=True, separators=(",", ":")))
PY
  )"
  /usr/bin/env bash -n "${fn_controller}"
  for fn_help in train-fold verify-inner-barrier evaluate-fold aggregate; do
    fn_token="$("${fn_python_bin}" -P -B "${fn_runtime}" "${fn_help}" --help)"
    grep -F -- --feature-root <<<"${fn_token}" >/dev/null || fail "${fn_help} help misses authority"
    grep -F -- --expected-feature-receipt-sha256 <<<"${fn_token}" >/dev/null || \
      fail "${fn_help} help misses expected feature SHA"
  done
  fn_help="$("${fn_python_bin}" -P -B "${fn_runtime}" train-fold --help)"
  for fn_token in --fold-index --fold-root --device; do
    grep -F -- "${fn_token}" <<<"${fn_help}" >/dev/null || fail "train help misses ${fn_token}"
  done
  fn_help="$("${fn_python_bin}" -P -B "${fn_runtime}" verify-inner-barrier --help)"
  for fn_token in --fold-root --expected-inner-receipt-sha256 --barrier-output --device; do
    grep -F -- "${fn_token}" <<<"${fn_help}" >/dev/null || fail "barrier help misses ${fn_token}"
  done
  fn_help="$("${fn_python_bin}" -P -B "${fn_runtime}" evaluate-fold --help)"
  for fn_token in --fold-index --barrier-receipt --expected-barrier-receipt-sha256 --device; do
    grep -F -- "${fn_token}" <<<"${fn_help}" >/dev/null || fail "evaluate help misses ${fn_token}"
  done
  grep -F -- --fold-root <<<"${fn_help}" >/dev/null && fail "evaluate CLI exposes child fold roots"
  grep -F -- --expected-inner-receipt-sha256 <<<"${fn_help}" >/dev/null && \
    fail "evaluate CLI exposes child inner SHAs"
  fn_help="$("${fn_python_bin}" -P -B "${fn_runtime}" aggregate --help)"
  for fn_token in --barrier-receipt --expected-barrier-receipt-sha256 \
    --expected-fold-receipt-sha256 --output; do
    grep -F -- "${fn_token}" <<<"${fn_help}" >/dev/null || fail "aggregate help misses ${fn_token}"
  done
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "preflight changed authority"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4G_PREFLIGHT_PASS={"job":%s,"node":"%s","cpus":%s,"memory":"4G","gres":"none","cpu_gate":%s,"normal_tests_run":36,"normal_tests_skipped":0,"optimized_tests_run":36,"optimized_tests_skipped":0,"tests_ran_in_parallel":true,"compile_opt0_and_opt2_passed":true,"ast_and_heredoc":%s,"bash_n_passed":true,"cli_help_passed":true,"controller_external_receipt_sha_arguments":false,"fresh_cache_cleaned":true}\n' \
    "${fn_job}" "${fn_node}" "${preflight_cpus}" "${fn_gate_json}" "${fn_heredoc_result}"
}

run_train_child() {
  [[ $# -eq 12 ]] || fail "train child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_receipt=$5 fn_v4d_receipt=$6 fn_fold_root=$7 fn_fold=$8
  local fn_job=$9 fn_node=${10} fn_controller_sha=${11} fn_snapshot=${12}
  local fn_controller fn_runtime fn_cache fn_now fn_result fn_verify
  fn_controller="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${worker_folds[${fn_fold}]}" == "${fn_fold}" \
      && "${worker_jobs[${fn_fold}]}" == "${fn_job}" \
      && "${worker_nodes[${fn_fold}]}" == "${fn_node}" \
      && "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "train holder mapping differs"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 train-controller
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "train initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "v4g-train-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}-fold${fn_fold}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  gpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" "train-fold${fn_fold}"
  consumer_wait_parent_directory "${fn_python_bin}" "$(dirname -- "${fn_fold_root}")" \
    0700 false "train-fold${fn_fold}-parent-run-root"
  [[ "${fn_fold_root}" == /* && ! -e "${fn_fold_root}" && ! -L "${fn_fold_root}" \
      && "$(basename -- "${fn_fold_root}")" == "fold${fn_fold}" ]] || \
    fail "train fold root freshness differs"
  mkdir "${fn_fold_root}"; chmod 0700 "${fn_fold_root}"
  if ! fn_result="$("${fn_python_bin}" -P -B "${fn_runtime}" train-fold \
      --feature-root "${fn_feature_root}" \
      --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
      --v4a-receipt "${fn_v4a_receipt}" \
      --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
      --v4c-frontier-receipt "${fn_v4c_receipt}" \
      --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
      --v4d-receipt "${fn_v4d_receipt}" \
      --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
      --fold-index "${fn_fold}" --fold-root "${fn_fold_root}" --device cuda:0)"; then
    fail "train-fold runtime failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_result}" "${fn_fold}" "${fn_fold_root}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if (value.get("fold_index") != int(sys.argv[2])
        or value.get("inner_receipt") != sys.argv[3] + "/inner.json"
        or value.get("oof_semantic_tensor_read_count") != 0
        or value.get("inference_authorized") is not False):
    raise SystemExit("train runtime stdout differs")
PY
  printf 'V4G_TRAIN_RUNTIME=%s\n' "${fn_result}"
  chmod 0555 "${fn_fold_root}"
  nfs_wait_path "${fn_fold_root}" directory "train-fold${fn_fold}-root"
  nfs_wait_path "${fn_fold_root}/inner.json" file "train-fold${fn_fold}-inner"
  fn_verify="$(verify_train_fold "${fn_python_bin}" "${fn_release_root}" \
    "${fn_fold_root}" "${fn_fold}")"
  [[ "${fn_verify}" == V4G_TRAIN_VERIFY=* ]] || fail "train verifier output differs"
  printf '%s\n' "${fn_verify}"
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "train changed authority"
  trap - EXIT; cleanup_cache "${fn_cache}"
  printf 'V4G_CACHE_CLEANED={"role":"train-fold%s","fresh_cache_root":"%s","absent_after_cleanup":true}\n' \
    "${fn_fold}" "${fn_cache}"
}

run_barrier_child() {
  [[ $# -eq 16 ]] || fail "barrier child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_receipt=$5 fn_v4d_receipt=$6 fn_run_root=$7 fn_job=$8 fn_node=$9
  local fn_controller_sha=${10} fn_snapshot=${11}
  local -a fn_train_bindings=("${12}" "${13}" "${14}" "${15}" "${16}")
  local -a fn_inner_shas=() fn_inner_sizes=() fn_pre_shas=() fn_pre_sizes=()
  local -a fn_fixed_shas=() fn_fixed_sizes=()
  local fn_controller fn_runtime fn_cache fn_now fn_result fn_verify fn_barrier fn_index
  fn_controller="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${worker_jobs[${barrier_worker_index}]}" \
      && "${fn_node}" == "${worker_nodes[${barrier_worker_index}]}" \
      && "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "barrier holder mapping differs"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 barrier-controller
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "barrier initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_barrier="${fn_run_root}/barrier/barrier.json"
  fn_cache="$(fresh_cache "v4g-barrier-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  gpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" verify-inner-barrier
  consumer_wait_parent_directory "${fn_python_bin}" "${fn_run_root}/barrier" \
    0700 true barrier-fresh-output-root
  for fn_index in 0 1 2 3 4; do
    IFS=: read -r fn_inner_shas[${fn_index}] fn_inner_sizes[${fn_index}] \
      fn_pre_shas[${fn_index}] fn_pre_sizes[${fn_index}] \
      fn_fixed_shas[${fn_index}] fn_fixed_sizes[${fn_index}] \
      <<<"${fn_train_bindings[${fn_index}]}"
    consumer_wait_directory "${fn_python_bin}" "${fn_run_root}/fold${fn_index}" \
      preselection.pt,fixed1200.pt,inner.json "barrier-fold${fn_index}-exact3"
    consumer_wait_file "${fn_python_bin}" "${fn_run_root}/fold${fn_index}/preselection.pt" \
      "${fn_pre_shas[${fn_index}]}" "${fn_pre_sizes[${fn_index}]}" \
      "barrier-fold${fn_index}-preselection"
    consumer_wait_file "${fn_python_bin}" "${fn_run_root}/fold${fn_index}/fixed1200.pt" \
      "${fn_fixed_shas[${fn_index}]}" "${fn_fixed_sizes[${fn_index}]}" \
      "barrier-fold${fn_index}-fixed1200"
    consumer_wait_file "${fn_python_bin}" "${fn_run_root}/fold${fn_index}/inner.json" \
      "${fn_inner_shas[${fn_index}]}" "${fn_inner_sizes[${fn_index}]}" \
      "barrier-fold${fn_index}-inner"
  done
  if ! fn_result="$("${fn_python_bin}" -P -B "${fn_runtime}" verify-inner-barrier \
      --feature-root "${fn_feature_root}" \
      --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
      --v4a-receipt "${fn_v4a_receipt}" \
      --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
      --v4c-frontier-receipt "${fn_v4c_receipt}" \
      --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
      --v4d-receipt "${fn_v4d_receipt}" \
      --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
      --fold-root "${fn_run_root}/fold0" --fold-root "${fn_run_root}/fold1" \
      --fold-root "${fn_run_root}/fold2" --fold-root "${fn_run_root}/fold3" \
      --fold-root "${fn_run_root}/fold4" \
      --expected-inner-receipt-sha256 "${fn_inner_shas[0]}" \
      --expected-inner-receipt-sha256 "${fn_inner_shas[1]}" \
      --expected-inner-receipt-sha256 "${fn_inner_shas[2]}" \
      --expected-inner-receipt-sha256 "${fn_inner_shas[3]}" \
      --expected-inner-receipt-sha256 "${fn_inner_shas[4]}" \
      --barrier-output "${fn_barrier}" --device cuda:0)"; then
    fail "verify-inner-barrier runtime failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_result}" "${fn_barrier}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if (value.get("barrier_receipt") != sys.argv[2]
        or value.get("all_five_inner_pass") is not True
        or value.get("oof_semantic_tensor_read_count") != 0
        or value.get("inference_authorized") is not False):
    raise SystemExit("barrier runtime stdout differs")
PY
  printf 'V4G_BARRIER_RUNTIME=%s\n' "${fn_result}"
  chmod 0555 "${fn_run_root}/barrier"
  nfs_wait_path "${fn_run_root}/barrier" directory barrier-root
  nfs_wait_path "${fn_barrier}" file barrier-receipt
  fn_verify="$(verify_barrier "${fn_python_bin}" "${fn_release_root}" \
    "${fn_run_root}" "${fn_barrier}" "${fn_inner_shas[@]}")"
  [[ "${fn_verify}" == V4G_BARRIER_VERIFY=* ]] || fail "barrier verifier output differs"
  printf '%s\n' "${fn_verify}"
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "barrier changed authority"
  trap - EXIT; cleanup_cache "${fn_cache}"
  printf 'V4G_CACHE_CLEANED={"role":"verify-inner-barrier","fresh_cache_root":"%s","absent_after_cleanup":true}\n' \
    "${fn_cache}"
}

run_evaluate_child() {
  [[ $# -eq 22 ]] || fail "evaluate child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_receipt=$5 fn_v4d_receipt=$6 fn_run_root=$7 fn_fold=$8
  local fn_barrier=$9 fn_barrier_sha=${10} fn_barrier_digest=${11} fn_inner_sha=${12}
  local fn_barrier_size=${13} fn_job=${14} fn_node=${15} fn_controller_sha=${16}
  local fn_snapshot=${17}
  local -a fn_train_bindings=("${18}" "${19}" "${20}" "${21}" "${22}")
  local -a fn_inner_shas=() fn_inner_sizes=() fn_pre_shas=() fn_pre_sizes=()
  local -a fn_fixed_shas=() fn_fixed_sizes=()
  local fn_controller fn_runtime fn_cache fn_now fn_result fn_verify fn_fold_root fn_index
  fn_controller="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${worker_jobs[${fn_fold}]}" == "${fn_job}" \
      && "${worker_nodes[${fn_fold}]}" == "${fn_node}" \
      && "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "evaluate holder mapping differs"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 evaluate-controller
  fn_fold_root="${fn_run_root}/fold${fn_fold}"
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "evaluate initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "v4g-evaluate-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}-fold${fn_fold}")"
  trap 'chmod 0555 "${fn_fold_root}" 2>/dev/null || true; cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  gpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" "evaluate-fold${fn_fold}"
  consumer_wait_directory "${fn_python_bin}" "${fn_run_root}/barrier" \
    barrier.json "evaluate-fold${fn_fold}-barrier-exact1"
  consumer_wait_file "${fn_python_bin}" "${fn_barrier}" "${fn_barrier_sha}" \
    "${fn_barrier_size}" "evaluate-fold${fn_fold}-barrier"
  consumer_wait_directory "${fn_python_bin}" "${fn_fold_root}" \
    preselection.pt,fixed1200.pt,inner.json \
    "evaluate-fold${fn_fold}-target-exact3"
  for fn_index in 0 1 2 3 4; do
    IFS=: read -r fn_inner_shas[${fn_index}] fn_inner_sizes[${fn_index}] \
      fn_pre_shas[${fn_index}] fn_pre_sizes[${fn_index}] \
      fn_fixed_shas[${fn_index}] fn_fixed_sizes[${fn_index}] \
      <<<"${fn_train_bindings[${fn_index}]}"
    consumer_wait_file "${fn_python_bin}" \
      "${fn_run_root}/fold${fn_index}/preselection.pt" \
      "${fn_pre_shas[${fn_index}]}" "${fn_pre_sizes[${fn_index}]}" \
      "evaluate-fold${fn_fold}-member-fold${fn_index}-preselection"
    consumer_wait_file "${fn_python_bin}" \
      "${fn_run_root}/fold${fn_index}/fixed1200.pt" \
      "${fn_fixed_shas[${fn_index}]}" "${fn_fixed_sizes[${fn_index}]}" \
      "evaluate-fold${fn_fold}-member-fold${fn_index}-fixed1200"
    consumer_wait_file "${fn_python_bin}" "${fn_run_root}/fold${fn_index}/inner.json" \
      "${fn_inner_shas[${fn_index}]}" "${fn_inner_sizes[${fn_index}]}" \
      "evaluate-fold${fn_fold}-member-fold${fn_index}-inner"
  done
  [[ "${fn_inner_sha}" == "${fn_inner_shas[${fn_fold}]}" ]] || \
    fail "evaluate target inner binding differs"
  [[ -d "${fn_fold_root}" && ! -L "${fn_fold_root}" \
      && "$(stat -c %a "${fn_fold_root}")" == 555 \
      && ! -e "${fn_fold_root}/fold.json" && ! -L "${fn_fold_root}/fold.json" ]] || \
    fail "evaluate fold root freshness differs"
  chmod 0700 "${fn_fold_root}"
  if ! fn_result="$("${fn_python_bin}" -P -B "${fn_runtime}" evaluate-fold \
      --feature-root "${fn_feature_root}" \
      --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
      --v4a-receipt "${fn_v4a_receipt}" \
      --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
      --v4c-frontier-receipt "${fn_v4c_receipt}" \
      --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
      --v4d-receipt "${fn_v4d_receipt}" \
      --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
      --fold-index "${fn_fold}" --barrier-receipt "${fn_barrier}" \
      --expected-barrier-receipt-sha256 "${fn_barrier_sha}" --device cuda:0)"; then
    fail "evaluate-fold runtime failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_result}" "${fn_fold}" "${fn_fold_root}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if (value.get("fold_index") != int(sys.argv[2])
        or value.get("fold_receipt") != sys.argv[3] + "/fold.json"
        or value.get("inference_authorized") is not False):
    raise SystemExit("evaluate runtime stdout differs")
PY
  printf 'V4G_EVALUATE_RUNTIME=%s\n' "${fn_result}"
  chmod 0555 "${fn_fold_root}"
  nfs_wait_path "${fn_fold_root}" directory "evaluate-fold${fn_fold}-root"
  nfs_wait_path "${fn_fold_root}/fold.json" file "evaluate-fold${fn_fold}-receipt"
  fn_verify="$(verify_evaluate_fold "${fn_python_bin}" "${fn_release_root}" \
    "${fn_fold_root}" "${fn_fold}" "${fn_barrier}" "${fn_barrier_sha}" \
    "${fn_barrier_digest}" "${fn_inner_sha}")"
  [[ "${fn_verify}" == V4G_EVALUATE_VERIFY=* ]] || fail "evaluate verifier output differs"
  printf '%s\n' "${fn_verify}"
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "evaluate changed authority"
  trap - EXIT; cleanup_cache "${fn_cache}"
  printf 'V4G_CACHE_CLEANED={"role":"evaluate-fold%s","fresh_cache_root":"%s","absent_after_cleanup":true}\n' \
    "${fn_fold}" "${fn_cache}"
}

run_aggregate_child() {
  [[ $# -eq 25 ]] || fail "aggregate child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_receipt=$5 fn_v4d_receipt=$6 fn_run_root=$7 fn_barrier=$8
  local fn_barrier_sha=$9 fn_barrier_digest=${10} fn_barrier_size=${11}
  local fn_job=${12} fn_node=${13} fn_controller_sha=${14} fn_snapshot=${15}
  local -a fn_fold_bindings=("${16}" "${17}" "${18}" "${19}" "${20}")
  local -a fn_train_bindings=("${21}" "${22}" "${23}" "${24}" "${25}")
  local -a fn_fold_shas=() fn_fold_sizes=()
  local -a fn_inner_shas=() fn_inner_sizes=() fn_pre_shas=() fn_pre_sizes=()
  local -a fn_fixed_shas=() fn_fixed_sizes=()
  local fn_controller fn_runtime fn_cache fn_now fn_result fn_verify fn_output fn_index
  fn_controller="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${cpu_job}" && "${fn_node}" == "${cpu_node}" \
      && "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "aggregate holder mapping differs"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 aggregate-controller
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "aggregate initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_output="${fn_run_root}/aggregate/receipt.json"
  fn_cache="$(fresh_cache "v4g-aggregate-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'chmod 0555 "${fn_run_root}/aggregate" 2>/dev/null || true; cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  cpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    aggregate v4g-vjepa2-cpu-aggregate-gate-v1 "${cpu_cpus}"
  consumer_wait_parent_directory "${fn_python_bin}" "${fn_run_root}/aggregate" \
    0700 true aggregate-fresh-output-root
  consumer_wait_directory "${fn_python_bin}" "${fn_run_root}/barrier" \
    barrier.json aggregate-barrier-exact1
  consumer_wait_file "${fn_python_bin}" "${fn_barrier}" "${fn_barrier_sha}" \
    "${fn_barrier_size}" aggregate-barrier
  for fn_index in 0 1 2 3 4; do
    IFS=: read -r fn_fold_shas[${fn_index}] fn_fold_sizes[${fn_index}] \
      <<<"${fn_fold_bindings[${fn_index}]}"
    IFS=: read -r fn_inner_shas[${fn_index}] fn_inner_sizes[${fn_index}] \
      fn_pre_shas[${fn_index}] fn_pre_sizes[${fn_index}] \
      fn_fixed_shas[${fn_index}] fn_fixed_sizes[${fn_index}] \
      <<<"${fn_train_bindings[${fn_index}]}"
    consumer_wait_directory "${fn_python_bin}" "${fn_run_root}/fold${fn_index}" \
      preselection.pt,fixed1200.pt,inner.json,fold.json \
      "aggregate-fold${fn_index}-exact4"
    consumer_wait_file "${fn_python_bin}" "${fn_run_root}/fold${fn_index}/fold.json" \
      "${fn_fold_shas[${fn_index}]}" "${fn_fold_sizes[${fn_index}]}" \
      "aggregate-fold${fn_index}-receipt"
    consumer_wait_file "${fn_python_bin}" \
      "${fn_run_root}/fold${fn_index}/preselection.pt" \
      "${fn_pre_shas[${fn_index}]}" "${fn_pre_sizes[${fn_index}]}" \
      "aggregate-fold${fn_index}-preselection"
    consumer_wait_file "${fn_python_bin}" \
      "${fn_run_root}/fold${fn_index}/fixed1200.pt" \
      "${fn_fixed_shas[${fn_index}]}" "${fn_fixed_sizes[${fn_index}]}" \
      "aggregate-fold${fn_index}-fixed1200"
    consumer_wait_file "${fn_python_bin}" "${fn_run_root}/fold${fn_index}/inner.json" \
      "${fn_inner_shas[${fn_index}]}" "${fn_inner_sizes[${fn_index}]}" \
      "aggregate-fold${fn_index}-inner"
  done
  if ! fn_result="$("${fn_python_bin}" -P -B "${fn_runtime}" aggregate \
      --feature-root "${fn_feature_root}" \
      --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
      --v4a-receipt "${fn_v4a_receipt}" \
      --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
      --v4c-frontier-receipt "${fn_v4c_receipt}" \
      --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
      --v4d-receipt "${fn_v4d_receipt}" \
      --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
      --barrier-receipt "${fn_barrier}" \
      --expected-barrier-receipt-sha256 "${fn_barrier_sha}" \
      --expected-fold-receipt-sha256 "${fn_fold_shas[0]}" \
      --expected-fold-receipt-sha256 "${fn_fold_shas[1]}" \
      --expected-fold-receipt-sha256 "${fn_fold_shas[2]}" \
      --expected-fold-receipt-sha256 "${fn_fold_shas[3]}" \
      --expected-fold-receipt-sha256 "${fn_fold_shas[4]}" \
      --output "${fn_output}")"; then
    fail "aggregate runtime failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_result}" "${fn_output}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if (value.get("receipt") != sys.argv[2]
        or type(value.get("exposed_five_view_codec_development_gate")) is not bool
        or value.get("inference_authorized") is not False):
    raise SystemExit("aggregate runtime stdout differs")
PY
  printf 'V4G_AGGREGATE_RUNTIME=%s\n' "${fn_result}"
  chmod 0555 "${fn_run_root}/aggregate"
  nfs_wait_path "${fn_run_root}/aggregate" directory aggregate-root
  nfs_wait_path "${fn_output}" file aggregate-receipt
  fn_verify="$(verify_aggregate "${fn_python_bin}" "${fn_release_root}" \
    "${fn_run_root}" "${fn_barrier_sha}" "${fn_barrier_digest}" \
    "${fn_fold_shas[@]}")"
  [[ "${fn_verify}" == V4G_AGGREGATE_VERIFY=* ]] || fail "aggregate verifier output differs"
  printf '%s\n' "${fn_verify}"
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "aggregate changed authority"
  trap - EXIT; cleanup_cache "${fn_cache}"
  printf 'V4G_CACHE_CLEANED={"role":"aggregate","fresh_cache_root":"%s","absent_after_cleanup":true}\n' \
    "${fn_cache}"
}

parent_stabilize_phase_inputs() {
  [[ $# -eq 5 || $# -eq 6 || $# -eq 10 ]] || \
    fail "parent_stabilize_phase_inputs argument count differs"
  local fn_python_bin=$1 fn_run_root=$2 fn_phase=$3
  local fn_history_encoded=$4 fn_history_sha=$5 fn_observation_mode=${6:-pre-seal}
  local fn_expected_seal_relative=${7:-none} fn_expected_seal_sha=${8:-none}
  local fn_expected_final_count=${9:-none} fn_expected_final_stable_tree_sha=${10:-none}
  "${fn_python_bin}" -I -S -B - "${fn_run_root}" "${fn_phase}" \
    "${fn_history_encoded}" "${fn_history_sha}" "${fn_observation_mode}" \
    "${fn_expected_seal_relative}" "${fn_expected_seal_sha}" \
    "${fn_expected_final_count}" "${fn_expected_final_stable_tree_sha}" <<'PY'
from pathlib import Path
import base64, binascii, hashlib, json, os, stat, sys, time

root = Path(sys.argv[1]); phase = sys.argv[2]
historical_encoded, historical_sha = sys.argv[3:5]
observation_mode = sys.argv[5]
expected_seal_relative, expected_seal_sha = sys.argv[6:8]
expected_final_count_text, expected_final_stable_tree_sha = sys.argv[8:10]
if observation_mode not in {"pre-seal", "post-seal"}:
    raise SystemExit("historical observation mode differs")
if observation_mode == "pre-seal" and sys.argv[6:10] != ["none"] * 4:
    raise SystemExit("pre-seal unexpected final-tree authority differs")
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
try:
    historical_raw = base64.b64decode(historical_encoded.encode("ascii"),
                                      altchars=b"-_", validate=True)
    historical = json.loads(historical_raw)
except (ValueError, binascii.Error, json.JSONDecodeError) as error:
    raise SystemExit("historical ledger decode differs: " + str(error))
train_history = historical.get("train_folds")
barrier_history = historical.get("barrier")
evaluate_history = historical.get("evaluate_folds")
aggregate_history = historical.get("aggregate")
history_counts = historical.get("captured_counts")
root_history_info = root.lstat()
expected_sequence = (["launch-plan"]
    + ["train-fold" + str(index) for index in range(len(train_history or []))]
    + (["verify-inner-barrier"] if barrier_history is not None else [])
    + ["evaluate-fold" + str(index) for index in range(len(evaluate_history or []))]
    + (["aggregate"] if aggregate_history is not None else []))
if (len(historical_encoded) > 98304
        or canonical(historical) != historical_raw
        or hashlib.sha256(historical_raw).hexdigest() != historical_sha
        or base64.b64encode(historical_raw, altchars=b"-_").decode("ascii")
           != historical_encoded
        or historical.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or historical.get("run_root") != str(root.resolve(strict=True))
        or historical.get("run_root_physical_identity") != {
            "device": root_history_info.st_dev, "inode": root_history_info.st_ino}
        or type(historical.get("launch_plan")) is not dict
        or type(train_history) is not list or type(evaluate_history) is not list
        or historical.get("append_sequence") != expected_sequence
        or [row.get("fold_index") for row in train_history]
           != list(range(len(train_history)))
        or [row.get("fold_index") for row in evaluate_history]
           != list(range(len(evaluate_history)))
        or history_counts != {
            "launch_plan": 1, "train": len(train_history),
            "barrier": int(barrier_history is not None),
            "evaluate_fold": len(evaluate_history),
            "aggregate": int(aggregate_history is not None)}):
    raise SystemExit("historical ledger prefix/count binding differs")
history_tuple = (len(train_history), int(barrier_history is not None),
                 len(evaluate_history), int(aggregate_history is not None))
history_phase_ok = {
    "TRAIN_FAILURE_ALL_OOF0": history_tuple == (0, 0, 0, 0),
    "TRAIN_VERIFICATION_FAILURE_ALL_OOF0": (
        0 <= history_tuple[0] < 5 and history_tuple[1:] == (0, 0, 0)),
    "INNER_NO_GO_ALL_OOF0": history_tuple == (5, 0, 0, 0),
    "BARRIER_FAILURE_ALL_OOF0": history_tuple == (5, 0, 0, 0),
    "EVALUATE_FAILURE": (history_tuple[0:2] == (5, 1)
                         and 0 <= history_tuple[2] < 5
                         and history_tuple[3] == 0),
    "AGGREGATE_FAILURE": history_tuple == (5, 1, 5, 0),
    "SUCCESS": history_tuple == (5, 1, 5, 1),
}.get(phase, False)
if not history_phase_ok:
    raise SystemExit("historical ledger phase-state count differs")
history_records = ([historical["launch_plan"]]
                   + train_history
                   + ([barrier_history] if barrier_history is not None else [])
                   + evaluate_history
                   + ([aggregate_history] if aggregate_history is not None else []))
historical_bindings = []
for record in history_records:
    if record is historical["launch_plan"]:
        historical_bindings.append(record.get("file_binding"))
    else:
        if type(record) is not dict or type(record.get("files")) is not list:
            raise SystemExit("historical stage record/file ledger differs")
        historical_bindings.extend(record["files"])
historical_by_path = {}
for row in historical_bindings:
    if type(row) is not dict:
        raise SystemExit("historical file binding row differs")
    relative = row.get("relative_path"); physical = row.get("physical_identity")
    expected_absolute = str(root / str(relative))
    if (not isinstance(relative, str) or relative.startswith("/") or ".." in relative.split("/")
            or row.get("absolute_path") != expected_absolute
            or row.get("mode_octal") != "0444" or row.get("nlink") != 1
            or row.get("same_fd_o_nofollow_verified") is not True
            or type(row.get("size_bytes")) is not int or row["size_bytes"] < 0
            or type(row.get("sha256")) is not str or len(row["sha256"]) != 64
            or type(physical) is not dict
            or physical.get("size_bytes") != row["size_bytes"]
            or type(physical.get("device")) is not int
            or type(physical.get("inode")) is not int
            or physical.get("mode_octal") != row["mode_octal"]
            or physical.get("nlink") != row["nlink"]
            or type(physical.get("mtime_ns")) is not int
            or type(physical.get("ctime_ns")) is not int):
        raise SystemExit("historical file binding envelope differs")
    if relative in historical_by_path and historical_by_path[relative] != row:
        raise SystemExit("historical repeated path binding changed")
    historical_by_path[relative] = row
if observation_mode == "post-seal":
    try:
        expected_final_count = int(expected_final_count_text)
    except ValueError:
        raise SystemExit("post-seal final count authority differs")
    if (expected_seal_relative not in {"seal.json", "global-no-go-seal.json"}
            or len(expected_seal_sha) != 64
            or len(expected_final_stable_tree_sha) != 64
            or expected_final_count <= 0):
        raise SystemExit("post-seal final-tree authority envelope differs")
    last = "not attempted"
    for attempt in range(1, 21):
        try:
            root_info = root.lstat()
            if (root.is_symlink() or not stat.S_ISDIR(root_info.st_mode)
                    or stat.S_IMODE(root_info.st_mode) != 0o555
                    or str(root) != str(root.resolve(strict=True))
                    or historical.get("run_root_physical_identity") != {
                        "device": root_info.st_dev, "inode": root_info.st_ino}):
                raise RuntimeError("post-seal run-root binding differs")
            members = list(root.rglob("*"))
            if any(path.is_symlink() or (not path.is_dir() and not path.is_file())
                   for path in members):
                raise RuntimeError("post-seal tree has symlink/special member")
            directories = sorted(path.relative_to(root).as_posix()
                                 for path in members if path.is_dir())
            if any(stat.S_IMODE((root / relative).lstat().st_mode) != 0o555
                   for relative in directories):
                raise RuntimeError("post-seal directory mode differs")
            stable_files = []
            parent_physical_by_path = {}
            for path in sorted(item for item in members if item.is_file()):
                relative = path.relative_to(root).as_posix(); before = path.lstat()
                if (path.is_symlink() or not stat.S_ISREG(before.st_mode)
                        or stat.S_IMODE(before.st_mode) != 0o444
                        or before.st_nlink != 1
                        or str(path) != str(path.resolve(strict=True))):
                    raise RuntimeError("post-seal file envelope differs: " + relative)
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                try:
                    opened = os.fstat(fd); digest = hashlib.sha256()
                    while True:
                        chunk = os.read(fd, 1024 * 1024)
                        if not chunk: break
                        digest.update(chunk)
                    closed = os.fstat(fd)
                finally:
                    os.close(fd)
                after = path.lstat()
                if not (identity(before) == identity(opened)
                        == identity(closed) == identity(after)):
                    raise RuntimeError(
                        "post-seal file same-FD identity differs: " + relative)
                stable_row = {
                    "path": relative, "sha256": digest.hexdigest(),
                    "size_bytes": before.st_size, "mode_octal": "0444", "nlink": 1}
                parent_physical = {
                    "device": before.st_dev, "inode": before.st_ino,
                    "size_bytes": before.st_size}
                expected_history = historical_by_path.get(relative)
                if (expected_history is not None
                        and (stable_row != {
                            "path": relative,
                            "sha256": expected_history["sha256"],
                            "size_bytes": expected_history["size_bytes"],
                            "mode_octal": expected_history["mode_octal"],
                            "nlink": expected_history["nlink"]}
                        or parent_physical != {
                            key: expected_history["physical_identity"][key]
                            for key in ("device", "inode", "size_bytes")})):
                    raise RuntimeError(
                        "post-seal historical binding differs: " + relative)
                stable_files.append(stable_row)
                parent_physical_by_path[relative] = parent_physical
            if (len(stable_files) != expected_final_count
                    or {row["path"] for row in stable_files}.intersection(
                        {"seal.json", "global-no-go-seal.json"})
                       != {expected_seal_relative}
                    or next(row for row in stable_files
                            if row["path"] == expected_seal_relative)["sha256"]
                       != expected_seal_sha
                    or object_sha({"directories": directories,
                                   "files": stable_files})
                       != expected_final_stable_tree_sha):
                raise RuntimeError("post-seal exact stable final tree differs")
            if not set(historical_by_path).issubset(parent_physical_by_path):
                raise RuntimeError("post-seal historical file membership differs")
            print("V4G_PARENT_HISTORICAL_POSTSEAL=" + json.dumps({
                "historical_ledger_sha256": historical_sha,
                "historical_exact_file_count": len(historical_by_path),
                "same_parent_root_and_file_physical_identity_reverified": True,
                "same_fd_sha_size_mode_nlink_reverified": True,
                "exact_final_tree_and_child_seal_sha_reverified": True,
                "final_stable_tree_sha256": expected_final_stable_tree_sha,
                "final_exact_file_count": expected_final_count,
                "attempt": attempt,
            }, sort_keys=True, separators=(",", ":")))
            raise SystemExit(0)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last = type(error).__name__ + ":" + str(error)
            print("V4G_PARENT_HISTORICAL_POSTSEAL_RETRY attempt=" + str(attempt)
                  + " diagnostic=" + last, file=sys.stderr, flush=True)
            if attempt != 20: time.sleep(1.0)
    raise SystemExit("parent historical post-seal timeout: " + last)
fold_dirs = {"fold" + str(index) for index in range(5)}
fold_train = {"fold" + str(index) + "/" + name for index in range(5)
              for name in ("preselection.pt", "fixed1200.pt", "inner.json")}
train_logs = {"logs/train-fold" + str(index) + suffix for index in range(5)
              for suffix in (".stdout", ".stderr")}
barrier_logs = {"logs/barrier.stdout", "logs/barrier.stderr"}
eval_logs = {"logs/evaluate-fold" + str(index) + suffix for index in range(5)
             for suffix in (".stdout", ".stderr")}
fold_receipts = {"fold" + str(index) + "/fold.json" for index in range(5)}
aggregate_logs = {"logs/aggregate.stdout", "logs/aggregate.stderr"}
base = {"launch-plan.json"} | train_logs; exact_train = base | fold_train
specs = {
    "TRAIN_FAILURE_ALL_OOF0": (base, fold_train, {"logs"}, {"logs"} | fold_dirs),
    "TRAIN_VERIFICATION_FAILURE_ALL_OOF0": (exact_train, set(),
        {"logs"} | fold_dirs, {"logs"} | fold_dirs),
    "INNER_NO_GO_ALL_OOF0": (exact_train, set(),
        {"logs"} | fold_dirs, {"logs"} | fold_dirs),
    "BARRIER_FAILURE_ALL_OOF0": (exact_train | barrier_logs,
        {"barrier/barrier.json"}, {"logs", "barrier"} | fold_dirs,
        {"logs", "barrier"} | fold_dirs),
    "EVALUATE_FAILURE": (exact_train | barrier_logs | {"barrier/barrier.json"}
        | eval_logs, fold_receipts, {"logs", "barrier"} | fold_dirs,
        {"logs", "barrier"} | fold_dirs),
    "AGGREGATE_FAILURE": (exact_train | barrier_logs | {"barrier/barrier.json"}
        | eval_logs | fold_receipts | aggregate_logs,
        {"aggregate/receipt.json"},
        {"logs", "barrier", "aggregate"} | fold_dirs,
        {"logs", "barrier", "aggregate"} | fold_dirs),
    "SUCCESS": (exact_train | barrier_logs | {"barrier/barrier.json"}
        | eval_logs | fold_receipts | aggregate_logs | {"aggregate/receipt.json"},
        set(), {"logs", "barrier", "aggregate"} | fold_dirs,
        {"logs", "barrier", "aggregate"} | fold_dirs),
}
if phase not in specs:
    raise SystemExit("parent stabilization phase differs")
required, optional, required_dirs, allowed_dirs = specs[phase]
def scan_once():
    if (not root.is_absolute() or root.is_symlink()
            or str(root) != str(root.resolve(strict=True))):
        raise RuntimeError("run root canonical/non-symlink differs")
    root_info = root.lstat()
    if (not stat.S_ISDIR(root_info.st_mode)
            or stat.S_IMODE(root_info.st_mode) != 0o700):
        raise RuntimeError("run root type/mode differs")
    members = list(root.rglob("*"))
    if any(path.is_symlink() or (not path.is_dir() and not path.is_file())
           for path in members):
        raise RuntimeError("tree has symlink/special member")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("directory no-follow support absent")
    for path in sorted((item for item in members if item.is_dir()), reverse=True):
        before = path.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise RuntimeError("directory type changed before normalization")
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC
                     | os.O_NOFOLLOW | os.O_DIRECTORY)
        try:
            opened = os.fstat(fd)
            if ((before.st_dev, before.st_ino, before.st_nlink)
                    != (opened.st_dev, opened.st_ino, opened.st_nlink)):
                raise RuntimeError("directory identity changed while opening")
            os.fchmod(fd, 0o700)
            closed = os.fstat(fd)
        finally:
            os.close(fd)
        after = path.lstat()
        if (not stat.S_ISDIR(closed.st_mode)
                or (opened.st_dev, opened.st_ino, opened.st_nlink)
                   != (closed.st_dev, closed.st_ino, closed.st_nlink)
                or (closed.st_dev, closed.st_ino, closed.st_nlink)
                   != (after.st_dev, after.st_ino, after.st_nlink)
                or stat.S_IMODE(closed.st_mode) != 0o700
                or stat.S_IMODE(after.st_mode) != 0o700):
            raise RuntimeError("directory same-FD normalization differs")
    members = list(root.rglob("*"))
    if any(path.is_symlink() or (not path.is_dir() and not path.is_file())
           for path in members):
        raise RuntimeError("tree changed to symlink/special after normalization")
    directories = sorted(({
        "path": path.relative_to(root).as_posix(),
        "mode_octal": format(stat.S_IMODE(path.lstat().st_mode), "04o"),
    } for path in members if path.is_dir()), key=lambda row: row["path"])
    directory_paths = {row["path"] for row in directories}
    if (not required_dirs.issubset(directory_paths)
            or not directory_paths.issubset(allowed_dirs)
            or any(row["mode_octal"] != "0700"
                   for row in directories)
            or phase != "TRAIN_FAILURE_ALL_OOF0" and directory_paths != allowed_dirs):
        raise RuntimeError("phase directory set/mode differs")
    rows = []; raws = {}
    for path in sorted(item for item in members if item.is_file()):
        relative = path.relative_to(root).as_posix(); before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1 or not hasattr(os, "O_NOFOLLOW")
                or str(path) != str(path.resolve(strict=True))):
            raise RuntimeError("file envelope differs: " + relative)
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk: break
                chunks.append(chunk); digest.update(chunk)
            closed = os.fstat(fd)
        finally:
            os.close(fd)
        after = path.lstat()
        if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
            raise RuntimeError("same-FD identity differs: " + relative)
        stable_row = {"path": relative, "sha256": digest.hexdigest(),
                      "size_bytes": before.st_size, "mode_octal": "0444",
                      "nlink": 1}
        historical_row = historical_by_path.get(relative)
        if historical_row is not None:
            if (stable_row != {
                    "path": historical_row["relative_path"],
                    "sha256": historical_row["sha256"],
                    "size_bytes": historical_row["size_bytes"],
                    "mode_octal": historical_row["mode_octal"],
                    "nlink": historical_row["nlink"]}
                    or historical_row["absolute_path"]
                       != str(path.resolve(strict=True))
                    or historical_row["physical_identity"] != {
                        "device": before.st_dev, "inode": before.st_ino,
                        "size_bytes": before.st_size,
                        "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
                        "nlink": before.st_nlink,
                        "mtime_ns": before.st_mtime_ns,
                        "ctime_ns": before.st_ctime_ns}):
                raise RuntimeError(
                    "historical same-parent full binding differs: " + relative)
        rows.append(stable_row)
        raws[relative] = b"".join(chunks)
    paths = {row["path"] for row in rows}
    if (not required.issubset(paths) or not paths.issubset(required | optional)
            or not set(historical_by_path).issubset(paths)):
        raise RuntimeError("phase file allowlist differs")
    return directories, rows, raws

last = None; stable = 0; final = None; diagnostic = "not attempted"
for attempt in range(1, 21):
    try:
        candidate = scan_once()
        signature = object_sha({"directories": candidate[0], "files": candidate[1]})
        stable = stable + 1 if signature == last else 1; last = signature; final = candidate
        diagnostic = "ok signature=" + signature + " consecutive=" + str(stable)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        stable = 0; last = None; final = None
        diagnostic = type(error).__name__ + ":" + str(error)
    print("V4G_PARENT_NFS_STABILIZE phase=" + phase + " attempt=" + str(attempt)
          + " diagnostic=" + diagnostic, file=sys.stderr, flush=True)
    if attempt != 20:
        time.sleep(1.0)
if final is None or stable < 3:
    raise SystemExit("parent phase inputs did not stabilize: " + diagnostic)
directories, rows, raws = final; by_path = {row["path"]: row for row in rows}
def exact_log(relative, prefixes):
    try:
        lines = raws[relative].decode("utf-8", errors="strict").splitlines()
        if (len(lines) != len(prefixes)
                or [line.split("=", 1)[0] for line in lines] != prefixes):
            return None
        return [json.loads(line.split("=", 1)[1]) for line in lines]
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError, IndexError):
        return None
def bound(relative, sha, size):
    row = by_path.get(relative)
    if row is None or row["sha256"] != sha or row["size_bytes"] != size:
        raise SystemExit("parent captured SHA/size differs: " + relative)
def train_capture(index):
    values = exact_log("logs/train-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME", "V4G_TRAIN_VERIFY",
         "V4G_CACHE_CLEANED"])
    if values is None: return None
    verify = values[2]
    if verify.get("fold_index") != index: raise SystemExit("parent train fold differs")
    bound("fold" + str(index) + "/inner.json",
          verify.get("inner_receipt_sha256"), verify.get("inner_receipt_size_bytes"))
    bound("fold" + str(index) + "/preselection.pt",
          verify.get("preselection_sha256"), verify.get("preselection_size_bytes"))
    bound("fold" + str(index) + "/fixed1200.pt",
          verify.get("fixed1200_sha256"), verify.get("fixed1200_size_bytes"))
    return {key: verify.get(key) for key in (
        "fold_index", "inner_pass", "inner_status", "inner_receipt_sha256",
        "inner_receipt_digest", "inner_receipt_size_bytes", "preselection_sha256",
        "preselection_size_bytes", "fixed1200_sha256", "fixed1200_size_bytes")}
trains = [value for value in (train_capture(index) for index in range(5))
          if value is not None]
barrier = None
values = exact_log("logs/barrier.stdout", ["V4G_GPU_GATE", "V4G_BARRIER_RUNTIME",
                   "V4G_BARRIER_VERIFY", "V4G_CACHE_CLEANED"])
if values is not None:
    verify = values[2]
    bound("barrier/barrier.json", verify.get("barrier_receipt_sha256"),
          verify.get("barrier_receipt_size_bytes"))
    barrier = {key: verify.get(key) for key in (
        "barrier_receipt_sha256", "barrier_receipt_digest",
        "barrier_receipt_size_bytes", "inner_receipt_shas")}
folds = []
for index in range(5):
    values = exact_log("logs/evaluate-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME", "V4G_EVALUATE_VERIFY",
         "V4G_CACHE_CLEANED"])
    if values is None: continue
    verify = values[2]
    bound("fold" + str(index) + "/fold.json", verify.get("fold_receipt_sha256"),
          verify.get("fold_receipt_size_bytes"))
    folds.append({key: verify.get(key) for key in (
        "fold_index", "fold_receipt_sha256", "fold_receipt_digest",
        "fold_receipt_size_bytes", "barrier_receipt_sha256",
        "inner_receipt_sha256")})
aggregate = None
values = exact_log("logs/aggregate.stdout", ["V4G_CPU_GATE", "V4G_AGGREGATE_RUNTIME",
                   "V4G_AGGREGATE_VERIFY", "V4G_CACHE_CLEANED"])
if values is not None:
    verify = values[2]
    bound("aggregate/receipt.json", verify.get("aggregate_receipt_sha256"),
          verify.get("aggregate_receipt_size_bytes"))
    aggregate = {key: verify.get(key) for key in (
        "aggregate_receipt_sha256", "aggregate_receipt_digest",
        "aggregate_receipt_size_bytes", "barrier_receipt_sha256",
        "fold_receipt_shas", "exposed_five_view_codec_development_gate")}
counts = {"train": len(trains), "barrier": int(barrier is not None),
          "evaluate_fold": len(folds), "aggregate": int(aggregate is not None)}
if phase in {"TRAIN_FAILURE_ALL_OOF0", "TRAIN_VERIFICATION_FAILURE_ALL_OOF0"}:
    ok = counts["barrier"] == counts["evaluate_fold"] == counts["aggregate"] == 0
elif phase == "INNER_NO_GO_ALL_OOF0":
    ok = counts == {"train": 5, "barrier": 0, "evaluate_fold": 0, "aggregate": 0}
elif phase == "BARRIER_FAILURE_ALL_OOF0":
    ok = (counts["train"] == 5 and counts["barrier"] in {0, 1}
          and counts["evaluate_fold"] == counts["aggregate"] == 0)
elif phase == "EVALUATE_FAILURE":
    ok = (counts["train"] == 5 and counts["barrier"] == 1
          and counts["aggregate"] == 0)
elif phase == "AGGREGATE_FAILURE":
    ok = (counts["train"] == 5 and counts["barrier"] == 1
          and counts["evaluate_fold"] == 5 and counts["aggregate"] in {0, 1})
else:
    ok = counts == {"train": 5, "barrier": 1, "evaluate_fold": 5, "aggregate": 1}
if not ok:
    raise SystemExit("parent captured stage count differs")
try:
    current_launch = json.loads(raws["launch-plan.json"])
except (KeyError, json.JSONDecodeError) as error:
    raise SystemExit("historical launch-plan decode differs: " + str(error))
if current_launch != historical["launch_plan"].get("value"):
    raise SystemExit("historical launch-plan full payload differs")
def receipt_value(relative):
    try:
        value = json.loads(raws[relative])
    except (KeyError, json.JSONDecodeError) as error:
        raise SystemExit("historical receipt decode differs: " + str(error))
    unsigned = dict(value); digest = unsigned.pop("receipt_digest", None)
    if digest != object_sha(unsigned):
        raise SystemExit("historical current receipt digest differs: " + relative)
    return value, digest
for record in train_history:
    index = record["fold_index"]
    values = exact_log("logs/train-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME", "V4G_TRAIN_VERIFY",
         "V4G_CACHE_CLEANED"])
    inner, digest = receipt_value("fold" + str(index) + "/inner.json")
    checkpoint_ledger = {
        "preselection_checkpoint_artifact": inner.get(
            "preselection_checkpoint_artifact"),
        "fixed1200_checkpoint_artifact": inner.get("fixed1200_checkpoint_artifact"),
        "preselection_fixed1200_checkpoint_pair_join": inner.get(
            "preselection_fixed1200_checkpoint_pair_join"),
        "fixed_candidate": inner.get("fixed_candidate"),
    }
    if (object_sha(values) != record.get("structured_values_sha256")
            or digest != record.get("receipt_digest")
            or inner.get("status") != record.get("receipt_status")
            or inner.get("inner_pass") is not record.get("inner_pass")
            or object_sha(checkpoint_ledger)
               != record.get("checkpoint_binding_ledger_sha256")):
        raise SystemExit("historical train semantic/full binding differs")
if barrier_history is not None:
    values = exact_log("logs/barrier.stdout", ["V4G_GPU_GATE",
        "V4G_BARRIER_RUNTIME", "V4G_BARRIER_VERIFY", "V4G_CACHE_CLEANED"])
    receipt, digest = receipt_value("barrier/barrier.json")
    if (object_sha(values)
            != barrier_history.get("structured_values_sha256")
            or digest != barrier_history.get("receipt_digest")
            or receipt.get("status") != barrier_history.get("receipt_status")
            or object_sha(receipt.get("members"))
               != barrier_history.get("full_member_binding_ledger_sha256")
            or object_sha(receipt.get("independent_replay_ledger"))
               != barrier_history.get("independent_replay_ledger_sha256")
            or receipt.get("independent_replay_sha256")
               != barrier_history.get("independent_replay_sha256")):
        raise SystemExit("historical barrier semantic/full binding differs")
for record in evaluate_history:
    index = record["fold_index"]
    values = exact_log("logs/evaluate-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME", "V4G_EVALUATE_VERIFY",
         "V4G_CACHE_CLEANED"])
    receipt, digest = receipt_value("fold" + str(index) + "/fold.json")
    global_barrier = receipt.get("global_inner_barrier")
    global_binding = ({
        "controller_barrier_receipt_binding": global_barrier.get(
            "controller_barrier_receipt_binding"),
        "inner_receipt_bindings": global_barrier.get("inner_receipt_bindings"),
        "independent_replay_ledger_sha256": object_sha(
            global_barrier.get("independent_replay_ledger")),
        "independent_replay_sha256": global_barrier.get(
            "independent_replay_sha256"),
    } if type(global_barrier) is dict else None)
    if (object_sha(values) != record.get("structured_values_sha256")
            or digest != record.get("receipt_digest")
            or receipt.get("status") != record.get("receipt_status")
            or receipt.get("inner_receipt_file_sha256")
               != record.get("inner_receipt_file_sha256")
            or receipt.get("controller_barrier_receipt_file_sha256")
               != record.get("controller_barrier_receipt_file_sha256")
            or object_sha(global_binding)
               != record.get("global_inner_barrier_binding_sha256")
            or object_sha(global_barrier)
               != record.get("global_inner_barrier_sha256")
            or object_sha(receipt.get("fixed1200_checkpoint_artifact"))
               != record.get("fixed1200_checkpoint_artifact_sha256")
            or object_sha(receipt.get("fixed1200_evaluate_checkpoint_binding"))
               != record.get("fixed1200_evaluate_checkpoint_binding_sha256")):
        raise SystemExit("historical evaluate semantic/full binding differs")
if aggregate_history is not None:
    values = exact_log("logs/aggregate.stdout", ["V4G_CPU_GATE",
        "V4G_AGGREGATE_RUNTIME", "V4G_AGGREGATE_VERIFY", "V4G_CACHE_CLEANED"])
    receipt, digest = receipt_value("aggregate/receipt.json")
    if (object_sha(values)
            != aggregate_history.get("structured_values_sha256")
            or digest != aggregate_history.get("receipt_digest")
            or receipt.get("status") != aggregate_history.get("receipt_status")
            or receipt.get("metrics", {}).get(
                "exposed_five_view_codec_development_gate")
               is not aggregate_history.get(
                   "exposed_five_view_codec_development_gate")
            or object_sha(receipt.get("controller_barrier_receipt_binding"))
               != aggregate_history.get(
                   "controller_barrier_receipt_binding_sha256")
            or object_sha(receipt.get("inner_receipts"))
               != aggregate_history.get("inner_receipt_binding_ledger_sha256")
            or object_sha(receipt.get("fold_receipts"))
               != aggregate_history.get("fold_receipt_binding_ledger_sha256")
            or object_sha(receipt.get("folds"))
               != aggregate_history.get("fold_value_ledger_sha256")
            or object_sha(receipt.get("global_inner_barrier"))
               != aggregate_history.get("global_inner_barrier_sha256")):
        raise SystemExit("historical aggregate semantic/full binding differs")
captured = {"train_folds": trains, "barrier": barrier,
            "evaluate_folds": folds, "aggregate": aggregate, "counts": counts}
transport = {"schema_version": "v4g-parent-stabilized-phase-inputs-v1",
             "phase": phase, "root_mode_octal": "0700",
             "directories": directories, "files": rows,
             "historical_ledger_sha256": historical_sha,
             "historical_captured_counts": history_counts,
             "historical_same_parent_physical_identity_reverified": True,
             "historical_semantic_full_binding_reverified": True,
             "captured_logical": captured,
             "captured_logical_sha256": object_sha(captured),
             "tree_sha256": object_sha({"directories": directories, "files": rows})}
raw = json.dumps(transport, sort_keys=True, separators=(",", ":"),
                 ensure_ascii=True, allow_nan=False).encode("ascii")
print(base64.urlsafe_b64encode(raw).decode("ascii") + ":" + object_sha(transport))
PY
}

consumer_wait_stabilized_phase_tree() {
  [[ $# -eq 5 ]] || fail "consumer_wait_stabilized_phase_tree argument count differs"
  local fn_python_bin=$1 fn_run_root=$2 fn_encoded=$3 fn_sha=$4 fn_phase=$5
  "${fn_python_bin}" -I -S -B - "${fn_run_root}" "${fn_encoded}" \
    "${fn_sha}" "${fn_phase}" <<'PY'
from pathlib import Path
import base64, hashlib, json, os, stat, sys, time
root = Path(sys.argv[1]); encoded, expected_sha, phase = sys.argv[2:5]
object_sha = lambda value: hashlib.sha256(json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).hexdigest()
try:
    value = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")))
except (ValueError, json.JSONDecodeError) as error:
    raise SystemExit("stabilized phase transport decode differs: " + str(error))
if (object_sha(value) != expected_sha or value.get("phase") != phase
        or value.get("schema_version") != "v4g-parent-stabilized-phase-inputs-v1"):
    raise SystemExit("stabilized phase transport binding differs")
expected_files = {row["path"]: row for row in value["files"]}
expected_dirs = {row["path"]: row for row in value["directories"]}
identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mode,
                         info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
last = "not attempted"
for attempt in range(1, 21):
    try:
        if (not root.is_absolute() or root.is_symlink()
                or str(root) != str(root.resolve(strict=True))
                or format(stat.S_IMODE(root.lstat().st_mode), "04o")
                   != value["root_mode_octal"]):
            raise RuntimeError("run root path/mode differs")
        members = list(root.rglob("*"))
        if any(path.is_symlink() or (not path.is_dir() and not path.is_file())
               for path in members):
            raise RuntimeError("phase tree has symlink/special member")
        directories = {path.relative_to(root).as_posix(): path
                       for path in members if path.is_dir()}
        files = {path.relative_to(root).as_posix(): path
                 for path in members if path.is_file()}
        if set(directories) != set(expected_dirs) or set(files) != set(expected_files):
            raise RuntimeError("phase exact membership differs")
        for relative, path in directories.items():
            if format(stat.S_IMODE(path.lstat().st_mode), "04o") != expected_dirs[relative]["mode_octal"]:
                raise RuntimeError("directory mode differs: " + relative)
        for relative, path in files.items():
            expected = expected_files[relative]; before = path.lstat()
            if (not stat.S_ISREG(before.st_mode)
                    or format(stat.S_IMODE(before.st_mode), "04o") != expected["mode_octal"]
                    or before.st_nlink != expected["nlink"]
                    or before.st_size != expected["size_bytes"]):
                raise RuntimeError("file envelope differs: " + relative)
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            try:
                opened = os.fstat(fd); digest = hashlib.sha256()
                while True:
                    chunk = os.read(fd, 1024 * 1024)
                    if not chunk: break
                    digest.update(chunk)
                closed = os.fstat(fd)
            finally:
                os.close(fd)
            after = path.lstat()
            if (not (identity(before) == identity(opened) == identity(closed)
                     == identity(after)) or digest.hexdigest() != expected["sha256"]):
                raise RuntimeError("file same-FD/SHA differs: " + relative)
        print("V4G_CONSUMER_NFS_VISIBLE label=seal-stabilized-phase-tree attempt="
              + str(attempt) + " files=" + str(len(files)) + " directories="
              + str(len(directories)), file=sys.stderr, flush=True)
        break
    except (FileNotFoundError, OSError, RuntimeError) as error:
        last = type(error).__name__ + ":" + str(error)
        print("V4G_CONSUMER_NFS_RETRY label=seal-stabilized-phase-tree attempt="
              + str(attempt) + " diagnostic=" + last, file=sys.stderr, flush=True)
        if attempt == 20:
            raise SystemExit("seal consumer stabilized-tree timeout: " + last)
        time.sleep(1.0)
PY
}

capture_phase_state() {
  [[ $# -eq 12 ]] || fail "capture_phase_state argument count differs"
  local fn_python_bin=$1 fn_run_root=$2 fn_phase=$3 fn_reason=$4
  local fn_allfold_oof0=$5 fn_snapshot=$6 fn_controller_sha=$7 fn_output=$8
  local fn_parent_encoded=$9 fn_parent_sha=${10}
  local fn_history_encoded=${11} fn_history_sha=${12}
  "${fn_python_bin}" -I -S -B - \
    "${fn_run_root}" "${fn_phase}" "${fn_reason}" "${fn_allfold_oof0}" \
    "${fn_snapshot}" "${fn_controller_sha}" "${fn_output}" \
    "${expected_release_tree_sha256}" "${expected_release_manifest_sha256}" \
    "${expected_release_manifest_digest}" "${expected_runtime_sha256}" \
    "${expected_runtime_test_sha256}" "${expected_python_sha256}" \
    "${expected_inner_schema}" "${expected_barrier_schema}" \
    "${expected_fold_schema}" "${expected_aggregate_schema}" \
    "${expected_inner_pass_status}" "${expected_inner_no_go_status}" \
    "${expected_barrier_pass_status}" "${expected_aggregate_status}" \
    "${fn_parent_encoded}" "${fn_parent_sha}" \
    "${fn_history_encoded}" "${fn_history_sha}" <<'PY'
from pathlib import Path
import base64, binascii, hashlib, json, os, stat, sys

run_root = Path(sys.argv[1]); phase, reason, allfold_text = sys.argv[2:5]
snapshot, controller_sha, output_text = sys.argv[5:8]
release_tree, manifest_sha, manifest_digest, runtime_sha, tests_sha, python_sha = sys.argv[8:14]
inner_schema, barrier_schema, fold_schema, aggregate_schema = sys.argv[14:18]
inner_pass_status, inner_no_go_status = sys.argv[18:20]
barrier_status, aggregate_status = sys.argv[20:22]
parent_encoded, parent_sha = sys.argv[22:24]
historical_encoded, historical_sha = sys.argv[24:26]
output = Path(output_text)
if (not run_root.is_absolute() or run_root.is_symlink()
        or str(run_root) != str(run_root.resolve(strict=True))
        or output != run_root / "phase-state.json"
        or output.exists() or output.is_symlink()
        or allfold_text not in {"true", "false"}):
    raise SystemExit("phase-state path/arguments differ")
allfold_oof0 = allfold_text == "true"
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)
def decode_json(raw, label):
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(label + " JSON differs: " + str(error))
try:
    parent_stable = json.loads(base64.urlsafe_b64decode(parent_encoded.encode("ascii")),
                               object_pairs_hook=pairs, parse_constant=nonfinite)
    historical_raw = base64.b64decode(historical_encoded.encode("ascii"),
                                      altchars=b"-_", validate=True)
    historical = json.loads(historical_raw, object_pairs_hook=pairs,
                            parse_constant=nonfinite)
except (ValueError, binascii.Error, json.JSONDecodeError) as error:
    raise SystemExit("parent/historical transport decode differs: " + str(error))
if (object_sha(parent_stable) != parent_sha
        or parent_stable.get("schema_version")
           != "v4g-parent-stabilized-phase-inputs-v1"
        or parent_stable.get("phase") != phase
        or len(historical_encoded) > 98304
        or canonical(historical) != historical_raw
        or hashlib.sha256(historical_raw).hexdigest() != historical_sha
        or base64.b64encode(historical_raw, altchars=b"-_").decode("ascii")
           != historical_encoded
        or historical.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or historical.get("run_root") != str(run_root)
        or type(historical.get("run_root_physical_identity")) is not dict
        or type(historical.get("run_root_physical_identity", {}).get("device")) is not int
        or type(historical.get("run_root_physical_identity", {}).get("inode")) is not int
        or parent_stable.get("historical_ledger_sha256") != historical_sha
        or parent_stable.get(
            "historical_same_parent_physical_identity_reverified") is not True
        or parent_stable.get(
            "historical_semantic_full_binding_reverified") is not True
        or parent_stable.get("historical_captured_counts")
           != historical.get("captured_counts")):
    raise SystemExit("parent stabilized/historical transport authority differs")
def bind(path):
    before = path.lstat()
    if (path.is_symlink() or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or not hasattr(os, "O_NOFOLLOW")):
        raise SystemExit("phase input file envelope differs: " + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit("phase input same-FD identity differs: " + str(path))
    raw = b"".join(chunks)
    return raw, {
        "path": path.relative_to(run_root).as_posix(),
        "sha256": digest.hexdigest(), "size_bytes": before.st_size,
        "mode_octal": "0444", "nlink": 1,
        "physical_identity": {"device": before.st_dev, "inode": before.st_ino,
                              "size_bytes": before.st_size,
                              "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
                              "nlink": before.st_nlink,
                              "mtime_ns": before.st_mtime_ns,
                              "ctime_ns": before.st_ctime_ns},
    }

members = list(run_root.rglob("*"))
if any(path.is_symlink() or (not path.is_dir() and not path.is_file()) for path in members):
    raise SystemExit("phase tree contains symlink or special member")
if stat.S_IMODE(run_root.lstat().st_mode) != 0o700:
    raise SystemExit("phase capture run-root mode differs")
actual_directories = {path.relative_to(run_root).as_posix()
                      for path in members if path.is_dir()}
if any(stat.S_IMODE((run_root / relative).lstat().st_mode) != 0o700
       for relative in actual_directories):
    raise SystemExit("phase capture directory mode differs")
files = []; raw_by_path = {}; binding_by_path = {}
for path in sorted(item for item in members if item.is_file()):
    raw, binding = bind(path); files.append(binding)
    raw_by_path[binding["path"]] = raw
    binding_by_path[binding["path"]] = binding
actual_files = set(raw_by_path)
stable_file_projection = [{key: row[key] for key in
                           ("path", "sha256", "size_bytes", "mode_octal", "nlink")}
                          for row in files]
if (stable_file_projection != parent_stable.get("files")
        or sorted(actual_directories)
           != sorted(row.get("path") for row in parent_stable.get("directories", []))):
    raise SystemExit("parent-to-seal-node stable tree logical binding differs")
train_history = historical.get("train_folds")
barrier_history = historical.get("barrier")
evaluate_history = historical.get("evaluate_folds")
aggregate_history = historical.get("aggregate")
history_counts = historical.get("captured_counts")
expected_sequence = (["launch-plan"]
    + ["train-fold" + str(index) for index in range(len(train_history or []))]
    + (["verify-inner-barrier"] if barrier_history is not None else [])
    + ["evaluate-fold" + str(index) for index in range(len(evaluate_history or []))]
    + (["aggregate"] if aggregate_history is not None else []))
if (type(historical.get("launch_plan")) is not dict
        or type(train_history) is not list or type(evaluate_history) is not list
        or historical.get("append_sequence") != expected_sequence
        or [row.get("fold_index") for row in train_history]
           != list(range(len(train_history)))
        or [row.get("fold_index") for row in evaluate_history]
           != list(range(len(evaluate_history)))
        or history_counts != {
            "launch_plan": 1, "train": len(train_history),
            "barrier": int(barrier_history is not None),
            "evaluate_fold": len(evaluate_history),
            "aggregate": int(aggregate_history is not None)}):
    raise SystemExit("historical ledger sequence/count differs on seal node")
history_records = ([historical["launch_plan"]] + train_history
                   + ([barrier_history] if barrier_history is not None else [])
                   + evaluate_history
                   + ([aggregate_history] if aggregate_history is not None else []))
historical_bindings = []
for record in history_records:
    if record is historical["launch_plan"]:
        historical_bindings.append(record.get("file_binding"))
    else:
        if type(record) is not dict or type(record.get("files")) is not list:
            raise SystemExit("historical stage record/file ledger differs on seal node")
        historical_bindings.extend(record["files"])
historical_stable_by_path = {}
for row in historical_bindings:
    if type(row) is not dict:
        raise SystemExit("historical file row differs on seal node")
    relative = row.get("relative_path"); current = binding_by_path.get(relative)
    stable = ({"path": row.get("relative_path"), "sha256": row.get("sha256"),
               "size_bytes": row.get("size_bytes"),
               "mode_octal": row.get("mode_octal"), "nlink": row.get("nlink")})
    if (not isinstance(relative, str) or current is None
            or row.get("absolute_path") != str(run_root / relative)
            or row.get("same_fd_o_nofollow_verified") is not True
            or stable != {key: current[key] for key in
                           ("path", "sha256", "size_bytes", "mode_octal", "nlink")}
            or relative in historical_stable_by_path
               and historical_stable_by_path[relative] != stable):
        raise SystemExit("historical cross-client stable file binding differs")
    historical_stable_by_path[relative] = stable
launch_history = historical["launch_plan"]
if decode_json(raw_by_path["launch-plan.json"], "historical launch plan") \
        != launch_history.get("value"):
    raise SystemExit("historical launch-plan full payload differs on seal node")
fold_dirs = {"fold" + str(index) for index in range(5)}
fold_train_files = {
    "fold" + str(index) + "/" + name
    for index in range(5)
    for name in ("preselection.pt", "fixed1200.pt", "inner.json")
}
train_logs = {
    "logs/train-fold" + str(index) + suffix
    for index in range(5) for suffix in (".stdout", ".stderr")
}
barrier_logs = {"logs/barrier.stdout", "logs/barrier.stderr"}
evaluate_logs = {
    "logs/evaluate-fold" + str(index) + suffix
    for index in range(5) for suffix in (".stdout", ".stderr")
}
fold_receipts = {"fold" + str(index) + "/fold.json" for index in range(5)}
aggregate_logs = {"logs/aggregate.stdout", "logs/aggregate.stderr"}
base = {"launch-plan.json"} | train_logs
exact_train = base | fold_train_files
contracts = {
    "TRAIN_FAILURE_ALL_OOF0": {
        "allfold": True, "required": base, "optional": fold_train_files,
        "required_dirs": {"logs"}, "allowed_dirs": {"logs"} | fold_dirs,
    },
    "TRAIN_VERIFICATION_FAILURE_ALL_OOF0": {
        "allfold": True, "required": exact_train, "optional": set(),
        "required_dirs": {"logs"} | fold_dirs,
        "allowed_dirs": {"logs"} | fold_dirs,
    },
    "INNER_NO_GO_ALL_OOF0": {
        "allfold": True, "required": exact_train, "optional": set(),
        "required_dirs": {"logs"} | fold_dirs,
        "allowed_dirs": {"logs"} | fold_dirs,
    },
    "BARRIER_FAILURE_ALL_OOF0": {
        "allfold": True, "required": exact_train | barrier_logs,
        "optional": {"barrier/barrier.json"},
        "required_dirs": {"logs", "barrier"} | fold_dirs,
        "allowed_dirs": {"logs", "barrier"} | fold_dirs,
    },
    "EVALUATE_FAILURE": {
        "allfold": False,
        "required": exact_train | barrier_logs | {"barrier/barrier.json"}
                    | evaluate_logs,
        "optional": fold_receipts,
        "required_dirs": {"logs", "barrier"} | fold_dirs,
        "allowed_dirs": {"logs", "barrier"} | fold_dirs,
    },
    "AGGREGATE_FAILURE": {
        "allfold": False,
        "required": exact_train | barrier_logs | {"barrier/barrier.json"}
                    | evaluate_logs | fold_receipts | aggregate_logs,
        "optional": {"aggregate/receipt.json"},
        "required_dirs": {"logs", "barrier", "aggregate"} | fold_dirs,
        "allowed_dirs": {"logs", "barrier", "aggregate"} | fold_dirs,
    },
    "SUCCESS": {
        "allfold": False,
        "required": exact_train | barrier_logs | {"barrier/barrier.json"}
                    | evaluate_logs | fold_receipts | aggregate_logs
                    | {"aggregate/receipt.json"},
        "optional": set(),
        "required_dirs": {"logs", "barrier", "aggregate"} | fold_dirs,
        "allowed_dirs": {"logs", "barrier", "aggregate"} | fold_dirs,
    },
}
contract = contracts.get(phase)
if contract is None or allfold_oof0 is not contract["allfold"]:
    raise SystemExit("phase/all-fold OOF0 contract differs")
if (not contract["required"].issubset(actual_files)
        or not actual_files.issubset(contract["required"] | contract["optional"])
        or not contract["required_dirs"].issubset(actual_directories)
        or not actual_directories.issubset(contract["allowed_dirs"])):
    raise SystemExit("phase-specific exact file/directory allowlist differs")
if phase == "TRAIN_FAILURE_ALL_OOF0":
    for index in range(5):
        directory = "fold" + str(index)
        present = {path for path in actual_files if path.startswith(directory + "/")}
        if (present and directory not in actual_directories
                or directory in actual_directories
                and not present.issubset({directory + "/" + name for name in
                    ("preselection.pt", "fixed1200.pt", "inner.json")})):
            raise SystemExit("train-failure partial fold tree differs")
else:
    if actual_directories != contract["allowed_dirs"]:
        raise SystemExit("phase-specific exact directory count differs")
launch = decode_json(raw_by_path["launch-plan.json"], "launch plan")
if (type(launch) is not dict
        or launch.get("schema_version") != "v4g-exact5-role-directed-launch-plan-v1"
        or launch.get("controller_sha256") != controller_sha
        or launch.get("authority_snapshot") != snapshot
        or launch.get("source_authority") != {
            "release_tree_sha256": release_tree,
            "release_manifest_sha256": manifest_sha,
            "release_manifest_digest": manifest_digest,
            "runtime_sha256": runtime_sha, "runtime_test_sha256": tests_sha,
            "python_sha256": python_sha,
        }
        or launch.get("fixed_no_fallback") is not True
        or launch.get("controller_captures_inner_barrier_and_fold_receipt_shas") is not True
        or launch.get("official_controller_cli_caller_supplied_inner_barrier_or_fold_sha") is not False):
    raise SystemExit("launch plan semantic authority differs")

def exact_log(relative, prefixes):
    try:
        lines = raw_by_path[relative].decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError:
        return None
    if len(lines) != len(prefixes) or [line.split("=", 1)[0] for line in lines] != prefixes:
        return None
    try:
        return [decode_json(line.split("=", 1)[1].encode("utf-8"), relative)
                for line in lines]
    except (IndexError, SystemExit):
        return None
def receipt(relative, label):
    value = decode_json(raw_by_path[relative], label)
    if type(value) is not dict:
        raise SystemExit(label + " root differs")
    unsigned = dict(value); digest = unsigned.pop("receipt_digest", None)
    if digest != object_sha(unsigned):
        raise SystemExit(label + " receipt digest differs")
    return value, digest
def require_file_binding(relative, sha, size, label):
    row = binding_by_path.get(relative)
    if (row is None or row.get("sha256") != sha
            or row.get("size_bytes") != size):
        raise SystemExit(label + " captured file binding differs")
    return dict(row)
checkpoint_binding_keys = (
    "path", "file_sha256", "size_bytes", "mode_octal", "nlink",
    "physical_identity", "single_fd_pre_post_sha256_exact",
    "semantic_metadata_state_replay_verified", "checkpoint_role", "outer_fold",
    "metadata_digest", "implementation_sha256", "model_state_sha256",
    "model_fit_original_count", "model_fit_ordered_iids", "model_fit_iid_digest",
    "inner_validation_iid_digest", "fixed_clip_pca_fit_input_sha256",
    "minibatch_schedule_sha256", "runtime_fingerprint",
    "model_schema_reconstructed_and_strict_loaded",
)
def checkpoint_projection(artifact):
    return {key: artifact.get(key) for key in checkpoint_binding_keys}
def artifact_base_exact(artifact, binding, absolute_path):
    return (type(artifact) is dict and artifact.get("path") == absolute_path
            and artifact.get("file_sha256") == binding["sha256"]
            and artifact.get("size_bytes") == binding["size_bytes"]
            and artifact.get("mode_octal") == binding["mode_octal"] == "0444"
            and artifact.get("nlink") == binding["nlink"] == 1
            and artifact.get("physical_identity") == binding["physical_identity"]
            and artifact.get("metadata_digest") is not None
            and artifact.get("model_state_sha256") is not None)

structured = []; train_captures = []
for index in range(5):
    relative = "logs/train-fold" + str(index) + ".stdout"
    values = exact_log(relative, ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME",
                                  "V4G_TRAIN_VERIFY", "V4G_CACHE_CLEANED"])
    if values is None:
        continue
    gate, runtime, verify, cache = values
    if (gate.get("role") != "train-fold" + str(index)
            or runtime.get("fold_index") != index
            or verify.get("fold_index") != index
            or runtime.get("inner_receipt_sha256") != verify.get("inner_receipt_sha256")
            or runtime.get("inner_receipt_digest") != verify.get("inner_receipt_digest")
            or runtime.get("inner_pass") is not verify.get("inner_pass")
            or verify.get("oof_semantic_tensor_read_count") != 0
            or cache.get("role") != "train-fold" + str(index)
            or cache.get("absent_after_cleanup") is not True):
        raise SystemExit("train captured stdout semantic join differs")
    inner_relative = "fold" + str(index) + "/inner.json"
    pre_relative = "fold" + str(index) + "/preselection.pt"
    fixed_relative = "fold" + str(index) + "/fixed1200.pt"
    inner_binding = require_file_binding(
        inner_relative, verify.get("inner_receipt_sha256"),
        verify.get("inner_receipt_size_bytes"), "inner")
    pre_binding = require_file_binding(
        pre_relative, verify.get("preselection_sha256"),
        verify.get("preselection_size_bytes"), "preselection")
    fixed_binding = require_file_binding(
        fixed_relative, verify.get("fixed1200_sha256"),
        verify.get("fixed1200_size_bytes"), "fixed1200")
    inner, inner_digest = receipt(inner_relative, "inner receipt")
    inner_pass = verify.get("inner_pass")
    expected_status = inner_pass_status if inner_pass is True else (
        inner_no_go_status if inner_pass is False else None)
    pre_artifact = inner.get("preselection_checkpoint_artifact")
    fixed_artifact = inner.get("fixed1200_checkpoint_artifact")
    pair = inner.get("preselection_fixed1200_checkpoint_pair_join")
    pre_absolute = str((run_root / pre_relative).resolve(strict=True))
    fixed_absolute = str((run_root / fixed_relative).resolve(strict=True))
    pre_subset = ({key: pre_artifact.get(key) for key in
                   ("path", "file_sha256", "size_bytes", "mode_octal", "nlink",
                    "metadata_digest", "model_state_sha256", "physical_identity")}
                  if type(pre_artifact) is dict else None)
    expected_pair = ({
        "preselection_path": pre_absolute, "fixed1200_path": fixed_absolute,
        "preselection_device_inode": [pre_binding["physical_identity"]["device"],
                                       pre_binding["physical_identity"]["inode"]],
        "fixed1200_device_inode": [fixed_binding["physical_identity"]["device"],
                                    fixed_binding["physical_identity"]["inode"]],
        "distinct_device_inode_pair": True, "same_model_state_sha256": True,
        "model_state_sha256": fixed_artifact.get("model_state_sha256"),
        "both_checkpoint_files_strongly_and_strictly_reloaded": True,
    } if type(fixed_artifact) is dict else None)
    provenance_fields = (
        "outer_fold", "model_state_sha256", "model_fit_original_count",
        "model_fit_ordered_iids", "model_fit_iid_digest",
        "inner_validation_iid_digest", "fixed_clip_pca_fit_input_sha256",
        "minibatch_schedule_sha256", "runtime_fingerprint", "implementation_sha256",
    )
    if (inner.get("schema_version") != inner_schema
            or inner.get("status") != expected_status
            or inner.get("fold_index") != index
            or inner.get("inner_pass") is not inner_pass
            or inner_digest != verify.get("inner_receipt_digest")
            or inner.get("oof_semantic_tensor_read_count_exact0") is not True
            or inner.get("oof_semantic_tensor_materialized_count") != 0
            or inner.get("oof_used_for_training_checkpoint_or_inner_gate") is not False
            or not artifact_base_exact(pre_artifact, pre_binding, pre_absolute)
            or not artifact_base_exact(fixed_artifact, fixed_binding, fixed_absolute)
            or pre_artifact.get("checkpoint_role") != "preselection_fixed_step1200"
            or fixed_artifact.get("checkpoint_role") != "fixed1200_candidate"
            or pre_artifact.get("outer_fold") != index
            or fixed_artifact.get("outer_fold") != index
            or any(pre_artifact.get(key) != fixed_artifact.get(key)
                   for key in provenance_fields)
            or fixed_artifact.get("preselection_checkpoint_file_sha256")
               != pre_artifact.get("file_sha256")
            or fixed_artifact.get("preselection_checkpoint_binding") != pre_subset
            or fixed_artifact.get("preselection_checkpoint_binding_sha256")
               != object_sha(pre_subset)
            or pair != expected_pair
            or pre_binding["physical_identity"] == fixed_binding["physical_identity"]
            or inner.get("qualification_scope", {}).get("inference_authorized") is not False):
        raise SystemExit("inner/checkpoint captured semantic join differs")
    inner_binding.update({"receipt_digest": inner_digest, "status": expected_status,
                          "inner_pass": inner_pass})
    pre_binding.update({"metadata_digest": pre_artifact.get("metadata_digest"),
                        "model_state_sha256": pre_artifact.get("model_state_sha256"),
                        "receipt_artifact": pre_artifact})
    fixed_binding.update({"metadata_digest": fixed_artifact.get("metadata_digest"),
                          "model_state_sha256": fixed_artifact.get("model_state_sha256"),
                          "receipt_artifact": fixed_artifact})
    loaded_pre = checkpoint_projection(pre_artifact)
    loaded_fixed = checkpoint_projection(fixed_artifact)
    inner_runtime_binding = {
        "fold_root": str((run_root / ("fold" + str(index))).resolve(strict=True)),
        "path": str((run_root / inner_relative).resolve(strict=True)),
        "file_sha256": inner_binding["sha256"], "receipt_digest": inner_digest,
        "mode_octal": "0444", "nlink": 1,
        "preselection_checkpoint_metadata_digest": pre_artifact["metadata_digest"],
        "fixed1200_checkpoint_metadata_digest": fixed_artifact["metadata_digest"],
        "checkpoint_provenance_binding_sha256": object_sha({
            "preselection": loaded_pre, "fixed1200": loaded_fixed}),
    }
    train_captures.append({"fold_index": index, "inner_receipt": inner_binding,
                           "preselection_checkpoint": pre_binding,
                           "fixed1200_checkpoint": fixed_binding,
                           "preselection_fixed1200_checkpoint_pair_join": pair,
                           "inner_runtime_binding": inner_runtime_binding,
                           "fixed_candidate_ledger_sha256": object_sha(
                               inner.get("fixed_candidate"))})
    structured.extend({"log": relative, "line": ordinal + 1,
                       "prefix": prefix, "value": value}
                      for ordinal, (prefix, value) in enumerate(zip(
                          ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME",
                           "V4G_TRAIN_VERIFY", "V4G_CACHE_CLEANED"], values)))

barrier_capture = None
if "logs/barrier.stdout" in actual_files:
    values = exact_log("logs/barrier.stdout", [
        "V4G_GPU_GATE", "V4G_BARRIER_RUNTIME", "V4G_BARRIER_VERIFY",
        "V4G_CACHE_CLEANED"])
    if values is not None:
        gate, runtime, verify, cache = values
        barrier_relative = "barrier/barrier.json"
        barrier_binding = require_file_binding(
            barrier_relative, verify.get("barrier_receipt_sha256"),
            verify.get("barrier_receipt_size_bytes"), "barrier")
        barrier, barrier_digest = receipt(barrier_relative, "barrier receipt")
        train_by_fold = {row["fold_index"]: row for row in train_captures}
        expected_inner_shas = [train_by_fold[index]["inner_receipt"]["sha256"]
                               for index in range(5)] if len(train_by_fold) == 5 else []
        members_value = barrier.get("members")
        replay_value = barrier.get("independent_replay_ledger")
        member_shas = ([row.get("inner_receipt_binding", {}).get("file_sha256")
                        for row in members_value]
                       if type(members_value) is list else None)
        expected_members = []
        if len(train_by_fold) == 5 and type(replay_value) is list and len(replay_value) == 5:
            required_replay_true = (
                "checkpoint_outer_fold_authority_join",
                "checkpoint_model_fit_ordered_iids_authority_join",
                "checkpoint_model_fit_count_and_digest_authority_join",
                "checkpoint_inner_iid_digest_authority_join",
                "checkpoint_pca_fit_input_receipt_training_join",
                "checkpoint_minibatch_schedule_receipt_training_join",
                "checkpoint_state_receipt_training_inner_join",
                "training_evaluate_runtime_fingerprint_exact_match",
                "checkpoint_clip_mean_equals_authority_recomputed_pca",
                "checkpoint_clip_basis_equals_authority_recomputed_pca",
                "checkpoint_fit_only_rms_equals_authority_recomputed_rms",
                "checkpoint_schema_and_exact79040_strict_loaded",
                "preselection_fixed1200_full_binding_reverified",
                "full1200_causal_weights_trusted_only_to_controller_pinned_sealed_training_execution",
                "authority_inner_five_views_re_materialized",
                "checkpoint_forward_reexecuted", "full_candidate_ledger_exact_match",
            )
            for replay_index, replay_row in enumerate(replay_value):
                train_row = train_by_fold[replay_index]
                inner_value, unused_digest = receipt(
                    "fold" + str(replay_index) + "/inner.json", "barrier inner receipt")
                candidate = inner_value.get("fixed_candidate")
                provenance = replay_row.get("model_fit_provenance_replay")
                inner_replay = replay_row.get("inner_replay_binding")
                training = inner_value.get("training")
                if (replay_row.get("fold_index") != replay_index
                        or replay_row.get("inner_receipt_file_sha256")
                           != train_row["inner_receipt"]["sha256"]
                        or replay_row.get("fixed1200_checkpoint_file_sha256")
                           != train_row["fixed1200_checkpoint"]["sha256"]
                        or replay_row.get("fixed1200_model_state_sha256")
                           != train_row["fixed1200_checkpoint"]["model_state_sha256"]
                        or type(provenance) is not dict
                        or replay_row.get("model_fit_provenance_replay_sha256")
                           != object_sha(provenance)
                        or provenance.get("fold_index") != replay_index
                        or provenance.get("model_fit_ordered_iids")
                           != inner_value.get("model_fit_ordered_iids")
                        or provenance.get("model_fit_iid_digest")
                           != inner_value.get("model_fit_iid_digest")
                        or provenance.get("model_fit_original_count")
                           != train_row["fixed1200_checkpoint"][
                               "receipt_artifact"].get("model_fit_original_count")
                        or provenance.get("clip_pca_fit_input_sha256")
                           != train_row["fixed1200_checkpoint"][
                               "receipt_artifact"].get("fixed_clip_pca_fit_input_sha256")
                        or type(training) is not dict
                        or provenance.get("minibatch_schedule_sha256")
                           != training.get("minibatch_schedule_sha256")
                        or type(candidate) is not dict or type(inner_replay) is not dict
                        or replay_row.get("inner_replay_sha256") != object_sha(inner_replay)
                        or inner_replay != {
                            "fold_index": replay_index,
                            "fixed_candidate_ledger_sha256": object_sha(candidate),
                            "inner_iid_digest": inner_value.get(
                                "inner_validation_iid_digest"),
                            "inner_evidence_sha256": candidate.get("inner_evidence_sha256"),
                            "complete_gate_sha256": object_sha(candidate.get("gate")),
                            "bootstrap_seed_ledger_sha256": object_sha(
                                candidate.get("bootstrap_seed_ledger")),
                            "fixed1200_checkpoint_file_sha256": train_row[
                                "fixed1200_checkpoint"]["sha256"],
                            "fixed1200_model_state_sha256": train_row[
                                "fixed1200_checkpoint"]["model_state_sha256"],
                        }
                        or any(replay_row.get(key) is not True
                               for key in required_replay_true)
                        or replay_row.get("full1200_optimizer_trajectory_reexecuted")
                           is not False
                        or replay_row.get("duplicate_training_performed") is not False
                        or replay_row.get("preselection_fixed1200_pair_join_sha256")
                           != object_sha(train_row[
                               "preselection_fixed1200_checkpoint_pair_join"])
                        or replay_row.get("inner_pass") is not True
                        or replay_row.get("oof_semantic_tensor_read_count") != 0):
                    raise SystemExit("barrier full provenance/replay binding differs")
                expected_members.append({
                    "fold_index": replay_index,
                    "fold_root": train_row["inner_runtime_binding"]["fold_root"],
                    "inner_receipt_binding": train_row["inner_runtime_binding"],
                    "preselection_checkpoint_artifact": train_row[
                        "preselection_checkpoint"]["receipt_artifact"],
                    "fixed1200_checkpoint_artifact": train_row[
                        "fixed1200_checkpoint"]["receipt_artifact"],
                    "preselection_fixed1200_checkpoint_pair_join": train_row[
                        "preselection_fixed1200_checkpoint_pair_join"],
                    "fixed_candidate_ledger_sha256": train_row[
                        "fixed_candidate_ledger_sha256"],
                    "independent_model_fit_provenance_replay_sha256": replay_row[
                        "model_fit_provenance_replay_sha256"],
                    "independent_inner_replay_sha256": replay_row[
                        "inner_replay_sha256"],
                    "inner_pass": True,
                    "oof_semantic_tensor_read_count_exact0": True,
                })
        if (gate.get("role") != "verify-inner-barrier"
                or runtime.get("barrier_receipt_sha256") != verify.get("barrier_receipt_sha256")
                or runtime.get("barrier_receipt_digest") != verify.get("barrier_receipt_digest")
                or runtime.get("oof_semantic_tensor_read_count") != 0
                or verify.get("oof_semantic_tensor_read_count") != 0
                or cache.get("role") != "verify-inner-barrier"
                or cache.get("absent_after_cleanup") is not True
                or barrier.get("schema_version") != barrier_schema
                or barrier.get("status") != barrier_status
                or barrier_digest != verify.get("barrier_receipt_digest")
                or type(members_value) is not list or len(members_value) != 5
                or [row.get("fold_index") for row in members_value] != list(range(5))
                or members_value != expected_members
                or barrier.get("members_sha256") != object_sha(members_value)
                or member_shas != expected_inner_shas
                or verify.get("inner_receipt_shas") != expected_inner_shas
                or type(replay_value) is not list or len(replay_value) != 5
                or barrier.get("independent_replay_sha256") != object_sha(replay_value)
                or barrier.get("all_five_exact_one_full_gates_pass") is not True
                or barrier.get("oof_semantic_tensor_read_count_exact0") is not True):
            raise SystemExit("barrier captured semantic join differs")
        barrier_binding.update({"receipt_digest": barrier_digest,
                                "status": barrier_status,
                                "member_inner_receipt_shas": member_shas,
                                "members_sha256": barrier.get("members_sha256"),
                                "members": members_value,
                                "independent_replay_ledger": replay_value,
                                "independent_replay_sha256": barrier.get(
                                    "independent_replay_sha256"),
                                "runtime_barrier_binding": {
                                    "path": str((run_root / barrier_relative).resolve(strict=True)),
                                    "file_sha256": barrier_binding["sha256"],
                                    "receipt_digest": barrier_digest,
                                    "mode_octal": "0444", "nlink": 1,
                                    "controller_expected_sha_exact": True,
                                }})
        barrier_capture = barrier_binding
        structured.extend({"log": "logs/barrier.stdout", "line": ordinal + 1,
                           "prefix": prefix, "value": value}
                          for ordinal, (prefix, value) in enumerate(zip(
                              ["V4G_GPU_GATE", "V4G_BARRIER_RUNTIME",
                               "V4G_BARRIER_VERIFY", "V4G_CACHE_CLEANED"], values)))

fold_captures = []
for index in range(5):
    relative = "logs/evaluate-fold" + str(index) + ".stdout"
    if relative not in actual_files:
        continue
    values = exact_log(relative, ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME",
                                  "V4G_EVALUATE_VERIFY", "V4G_CACHE_CLEANED"])
    if values is None:
        continue
    if barrier_capture is None or len(train_captures) != 5:
        raise SystemExit("evaluate capture lacks train/barrier authority")
    gate, runtime, verify, cache = values
    fold_relative = "fold" + str(index) + "/fold.json"
    fold_binding = require_file_binding(
        fold_relative, verify.get("fold_receipt_sha256"),
        verify.get("fold_receipt_size_bytes"), "fold receipt")
    fold_value, fold_digest = receipt(fold_relative, "fold receipt")
    expected_inner_sha = train_captures[index]["inner_receipt"]["sha256"]
    global_barrier = fold_value.get("global_inner_barrier")
    expected_fixed_artifact = train_captures[index][
        "fixed1200_checkpoint"]["receipt_artifact"]
    expected_current_fixed_binding = checkpoint_projection(expected_fixed_artifact)
    expected_inner_runtime_bindings = [row["inner_runtime_binding"]
                                       for row in train_captures]
    expected_barrier_binding = barrier_capture["runtime_barrier_binding"]
    if (gate.get("role") != "evaluate-fold" + str(index)
            or runtime.get("fold_index") != index or verify.get("fold_index") != index
            or runtime.get("fold_receipt_sha256") != verify.get("fold_receipt_sha256")
            or runtime.get("fold_receipt_digest") != verify.get("fold_receipt_digest")
            or verify.get("barrier_receipt_sha256") != barrier_capture["sha256"]
            or verify.get("inner_receipt_sha256") != expected_inner_sha
            or cache.get("role") != "evaluate-fold" + str(index)
            or cache.get("absent_after_cleanup") is not True
            or fold_value.get("schema_version") != fold_schema
            or fold_value.get("status") != aggregate_status
            or fold_value.get("fold_index") != index
            or fold_digest != verify.get("fold_receipt_digest")
            or fold_value.get("inner_receipt_file_sha256") != expected_inner_sha
            or fold_value.get("controller_barrier_receipt_file_sha256")
               != barrier_capture["sha256"]
            or type(global_barrier) is not dict
            or global_barrier.get("controller_barrier_receipt_digest")
               != barrier_capture["receipt_digest"]
            or global_barrier.get("controller_barrier_receipt_binding")
               != expected_barrier_binding
            or global_barrier.get("inner_receipt_bindings")
               != expected_inner_runtime_bindings
            or global_barrier.get("independent_replay_ledger")
               != barrier_capture["independent_replay_ledger"]
            or global_barrier.get("independent_replay_sha256")
               != barrier_capture["independent_replay_sha256"]
            or fold_value.get("fixed1200_checkpoint_artifact")
               != expected_fixed_artifact
            or fold_value.get("fixed1200_evaluate_checkpoint_binding")
               != expected_current_fixed_binding
            or fold_value.get("oof_used_for_training_checkpoint_inner_gate_or_selection")
               is not False
            or fold_value.get("qualification_scope", {}).get("inference_authorized")
               is not False):
        raise SystemExit("evaluate captured semantic join differs")
    fold_binding.update({"receipt_digest": fold_digest, "status": aggregate_status,
                         "fold_index": index,
                         "barrier_receipt_sha256": barrier_capture["sha256"],
                         "inner_receipt_sha256": expected_inner_sha,
                         "current_fixed1200_binding": expected_current_fixed_binding,
                         "runtime_fold_binding": {
                             "fold_root": str((run_root / ("fold" + str(index))).resolve(
                                 strict=True)),
                             "path": str((run_root / fold_relative).resolve(strict=True)),
                             "file_sha256": fold_binding["sha256"],
                             "receipt_digest": fold_digest,
                             "mode_octal": "0444", "nlink": 1,
                         }})
    fold_captures.append(fold_binding)
    structured.extend({"log": relative, "line": ordinal + 1,
                       "prefix": prefix, "value": value}
                      for ordinal, (prefix, value) in enumerate(zip(
                          ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME",
                           "V4G_EVALUATE_VERIFY", "V4G_CACHE_CLEANED"], values)))

aggregate_capture = None
if "logs/aggregate.stdout" in actual_files:
    values = exact_log("logs/aggregate.stdout", [
        "V4G_CPU_GATE", "V4G_AGGREGATE_RUNTIME", "V4G_AGGREGATE_VERIFY",
        "V4G_CACHE_CLEANED"])
    if values is not None:
        if barrier_capture is None or len(fold_captures) != 5:
            raise SystemExit("aggregate capture lacks barrier/fold authority")
        gate, runtime, verify, cache = values
        aggregate_relative = "aggregate/receipt.json"
        aggregate_binding = require_file_binding(
            aggregate_relative, verify.get("aggregate_receipt_sha256"),
            verify.get("aggregate_receipt_size_bytes"), "aggregate receipt")
        aggregate, aggregate_digest = receipt(aggregate_relative, "aggregate receipt")
        fold_shas = [row["sha256"] for row in fold_captures]
        fold_runtime_bindings = [row["runtime_fold_binding"] for row in fold_captures]
        inner_runtime_bindings = [row["inner_runtime_binding"]
                                  for row in train_captures]
        fold_values = [receipt("fold" + str(index) + "/fold.json",
                               "aggregate fold receipt")[0] for index in range(5)]
        gate_value = verify.get("exposed_five_view_codec_development_gate")
        if (gate.get("role") != "aggregate"
                or runtime.get("receipt_sha256") != verify.get("aggregate_receipt_sha256")
                or runtime.get("receipt_digest") != verify.get("aggregate_receipt_digest")
                or runtime.get("exposed_five_view_codec_development_gate") is not gate_value
                or verify.get("barrier_receipt_sha256") != barrier_capture["sha256"]
                or verify.get("fold_receipt_shas") != fold_shas
                or cache.get("role") != "aggregate"
                or cache.get("absent_after_cleanup") is not True
                or aggregate.get("schema_version") != aggregate_schema
                or aggregate.get("status") != aggregate_status
                or aggregate_digest != verify.get("aggregate_receipt_digest")
                or aggregate.get("controller_barrier_receipt_binding")
                   != barrier_capture["runtime_barrier_binding"]
                or aggregate.get("controller_barrier_receipt_digest")
                   != barrier_capture["receipt_digest"]
                or aggregate.get("inner_receipts") != {
                    "count": 5, "bindings": inner_runtime_bindings}
                or aggregate.get("fold_receipts") != {
                    "count": 5, "bindings": fold_runtime_bindings}
                or aggregate.get("folds") != fold_values
                or aggregate.get("global_inner_barrier", {}).get(
                    "inner_receipt_bindings") != inner_runtime_bindings
                or aggregate.get("global_inner_barrier", {}).get(
                    "common_independent_replay_ledger")
                   != barrier_capture["independent_replay_ledger"]
                or aggregate.get("global_inner_barrier", {}).get(
                    "common_independent_replay_sha256")
                   != barrier_capture["independent_replay_sha256"]
                or aggregate.get("metrics", {}).get(
                    "exposed_five_view_codec_development_gate") is not gate_value
                or type(gate_value) is not bool
                or aggregate.get("qualification_scope", {}).get(
                    "aggregate_gate_evaluated") is not True
                or aggregate.get("qualification_scope", {}).get(
                    "inference_authorized") is not False):
            raise SystemExit("aggregate captured semantic join differs")
        aggregate_binding.update({"receipt_digest": aggregate_digest,
                                  "status": aggregate_status,
                                  "barrier_receipt_sha256": barrier_capture["sha256"],
                                  "fold_receipt_shas": fold_shas,
                                  "runtime_barrier_binding": barrier_capture[
                                      "runtime_barrier_binding"],
                                  "runtime_inner_receipt_bindings": inner_runtime_bindings,
                                  "runtime_fold_receipt_bindings": fold_runtime_bindings,
                                  "exposed_five_view_codec_development_gate": gate_value,
                                  "inference_authorized": False})
        aggregate_capture = aggregate_binding
        structured.extend({"log": "logs/aggregate.stdout", "line": ordinal + 1,
                           "prefix": prefix, "value": value}
                          for ordinal, (prefix, value) in enumerate(zip(
                              ["V4G_CPU_GATE", "V4G_AGGREGATE_RUNTIME",
                               "V4G_AGGREGATE_VERIFY", "V4G_CACHE_CLEANED"], values)))

counts = {"train": len(train_captures), "barrier": int(barrier_capture is not None),
          "evaluate_fold": len(fold_captures),
          "aggregate": int(aggregate_capture is not None)}
passes = [row["inner_receipt"]["inner_pass"] for row in train_captures]
if phase in {"TRAIN_FAILURE_ALL_OOF0", "TRAIN_VERIFICATION_FAILURE_ALL_OOF0"}:
    counts_ok = (0 <= counts["train"] <= 5 and counts["barrier"] == 0
                 and counts["evaluate_fold"] == 0 and counts["aggregate"] == 0)
elif phase == "INNER_NO_GO_ALL_OOF0":
    counts_ok = (counts == {"train": 5, "barrier": 0,
                            "evaluate_fold": 0, "aggregate": 0}
                 and any(value is False for value in passes))
elif phase == "BARRIER_FAILURE_ALL_OOF0":
    counts_ok = (counts["train"] == 5 and all(value is True for value in passes)
                 and counts["barrier"] in {0, 1}
                 and counts["evaluate_fold"] == 0 and counts["aggregate"] == 0)
elif phase == "EVALUATE_FAILURE":
    counts_ok = (counts["train"] == 5 and all(value is True for value in passes)
                 and counts["barrier"] == 1
                 and 0 <= counts["evaluate_fold"] <= 5 and counts["aggregate"] == 0)
elif phase == "AGGREGATE_FAILURE":
    counts_ok = (counts["train"] == 5 and all(value is True for value in passes)
                 and counts["barrier"] == 1 and counts["evaluate_fold"] == 5
                 and counts["aggregate"] in {0, 1})
else:
    counts_ok = (counts == {"train": 5, "barrier": 1,
                            "evaluate_fold": 5, "aggregate": 1}
                 and all(value is True for value in passes))
if not counts_ok:
    raise SystemExit("phase-specific captured receipt count/binding state differs")
captured = {"train_folds": train_captures, "barrier": barrier_capture,
            "evaluate_folds": fold_captures, "aggregate": aggregate_capture,
            "counts": counts}
logical = {
    "train_folds": [{
        "fold_index": row["fold_index"],
        "inner_pass": row["inner_receipt"]["inner_pass"],
        "inner_status": row["inner_receipt"]["status"],
        "inner_receipt_sha256": row["inner_receipt"]["sha256"],
        "inner_receipt_digest": row["inner_receipt"]["receipt_digest"],
        "inner_receipt_size_bytes": row["inner_receipt"]["size_bytes"],
        "preselection_sha256": row["preselection_checkpoint"]["sha256"],
        "preselection_size_bytes": row["preselection_checkpoint"]["size_bytes"],
        "fixed1200_sha256": row["fixed1200_checkpoint"]["sha256"],
        "fixed1200_size_bytes": row["fixed1200_checkpoint"]["size_bytes"],
    } for row in train_captures],
    "barrier": (None if barrier_capture is None else {
        "barrier_receipt_sha256": barrier_capture["sha256"],
        "barrier_receipt_digest": barrier_capture["receipt_digest"],
        "barrier_receipt_size_bytes": barrier_capture["size_bytes"],
        "inner_receipt_shas": barrier_capture["member_inner_receipt_shas"],
    }),
    "evaluate_folds": [{
        "fold_index": row["fold_index"],
        "fold_receipt_sha256": row["sha256"],
        "fold_receipt_digest": row["receipt_digest"],
        "fold_receipt_size_bytes": row["size_bytes"],
        "barrier_receipt_sha256": row["barrier_receipt_sha256"],
        "inner_receipt_sha256": row["inner_receipt_sha256"],
    } for row in fold_captures],
    "aggregate": (None if aggregate_capture is None else {
        "aggregate_receipt_sha256": aggregate_capture["sha256"],
        "aggregate_receipt_digest": aggregate_capture["receipt_digest"],
        "aggregate_receipt_size_bytes": aggregate_capture["size_bytes"],
        "barrier_receipt_sha256": aggregate_capture["barrier_receipt_sha256"],
        "fold_receipt_shas": aggregate_capture["fold_receipt_shas"],
        "exposed_five_view_codec_development_gate": aggregate_capture[
            "exposed_five_view_codec_development_gate"],
    }),
    "counts": counts,
}
if (logical != parent_stable.get("captured_logical")
        or object_sha(logical) != parent_stable.get("captured_logical_sha256")):
    raise SystemExit("parent-to-seal-node captured logical bindings differ")
history_tuple = (len(train_history), int(barrier_history is not None),
                 len(evaluate_history), int(aggregate_history is not None))
history_phase_ok = {
    "TRAIN_FAILURE_ALL_OOF0": history_tuple == (0, 0, 0, 0),
    "TRAIN_VERIFICATION_FAILURE_ALL_OOF0": (
        0 <= history_tuple[0] < 5 and history_tuple[1:] == (0, 0, 0)),
    "INNER_NO_GO_ALL_OOF0": history_tuple == (5, 0, 0, 0),
    "BARRIER_FAILURE_ALL_OOF0": history_tuple == (5, 0, 0, 0),
    "EVALUATE_FAILURE": (history_tuple[0:2] == (5, 1)
                         and 0 <= history_tuple[2] < 5
                         and history_tuple[3] == 0),
    "AGGREGATE_FAILURE": history_tuple == (5, 1, 5, 0),
    "SUCCESS": history_tuple == (5, 1, 5, 1),
}.get(phase, False)
if (not history_phase_ok or len(train_history) > len(train_captures)
        or len(evaluate_history) > len(fold_captures)
        or int(barrier_history is not None) > int(barrier_capture is not None)
        or int(aggregate_history is not None) > int(aggregate_capture is not None)):
    raise SystemExit("historical ledger phase/current stage count differs")
def historical_receipt(relative):
    value = decode_json(raw_by_path[relative], "historical receipt")
    unsigned = dict(value); digest = unsigned.pop("receipt_digest", None)
    if digest != object_sha(unsigned):
        raise SystemExit("historical current receipt digest differs")
    return value, digest
for record in train_history:
    index = record["fold_index"]
    values = exact_log("logs/train-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME", "V4G_TRAIN_VERIFY",
         "V4G_CACHE_CLEANED"])
    inner, digest = historical_receipt("fold" + str(index) + "/inner.json")
    checkpoint_ledger = {
        "preselection_checkpoint_artifact": inner.get(
            "preselection_checkpoint_artifact"),
        "fixed1200_checkpoint_artifact": inner.get("fixed1200_checkpoint_artifact"),
        "preselection_fixed1200_checkpoint_pair_join": inner.get(
            "preselection_fixed1200_checkpoint_pair_join"),
        "fixed_candidate": inner.get("fixed_candidate"),
    }
    if (object_sha(values) != record.get("structured_values_sha256")
            or digest != record.get("receipt_digest")
            or inner.get("status") != record.get("receipt_status")
            or inner.get("inner_pass") is not record.get("inner_pass")
            or object_sha(checkpoint_ledger)
               != record.get("checkpoint_binding_ledger_sha256")):
        raise SystemExit("historical train semantic/full binding differs on seal node")
if barrier_history is not None:
    values = exact_log("logs/barrier.stdout", ["V4G_GPU_GATE",
        "V4G_BARRIER_RUNTIME", "V4G_BARRIER_VERIFY", "V4G_CACHE_CLEANED"])
    receipt, digest = historical_receipt("barrier/barrier.json")
    if (object_sha(values)
            != barrier_history.get("structured_values_sha256")
            or digest != barrier_history.get("receipt_digest")
            or receipt.get("status") != barrier_history.get("receipt_status")
            or object_sha(receipt.get("members"))
               != barrier_history.get("full_member_binding_ledger_sha256")
            or object_sha(receipt.get("independent_replay_ledger"))
               != barrier_history.get("independent_replay_ledger_sha256")
            or receipt.get("independent_replay_sha256")
               != barrier_history.get("independent_replay_sha256")):
        raise SystemExit("historical barrier semantic/full binding differs on seal node")
for record in evaluate_history:
    index = record["fold_index"]
    values = exact_log("logs/evaluate-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME", "V4G_EVALUATE_VERIFY",
         "V4G_CACHE_CLEANED"])
    receipt, digest = historical_receipt("fold" + str(index) + "/fold.json")
    global_barrier = receipt.get("global_inner_barrier")
    global_binding = ({
        "controller_barrier_receipt_binding": global_barrier.get(
            "controller_barrier_receipt_binding"),
        "inner_receipt_bindings": global_barrier.get("inner_receipt_bindings"),
        "independent_replay_ledger_sha256": object_sha(
            global_barrier.get("independent_replay_ledger")),
        "independent_replay_sha256": global_barrier.get(
            "independent_replay_sha256"),
    } if type(global_barrier) is dict else None)
    if (object_sha(values) != record.get("structured_values_sha256")
            or digest != record.get("receipt_digest")
            or receipt.get("status") != record.get("receipt_status")
            or receipt.get("inner_receipt_file_sha256")
               != record.get("inner_receipt_file_sha256")
            or receipt.get("controller_barrier_receipt_file_sha256")
               != record.get("controller_barrier_receipt_file_sha256")
            or object_sha(global_binding)
               != record.get("global_inner_barrier_binding_sha256")
            or object_sha(global_barrier)
               != record.get("global_inner_barrier_sha256")
            or object_sha(receipt.get("fixed1200_checkpoint_artifact"))
               != record.get("fixed1200_checkpoint_artifact_sha256")
            or object_sha(receipt.get("fixed1200_evaluate_checkpoint_binding"))
               != record.get("fixed1200_evaluate_checkpoint_binding_sha256")):
        raise SystemExit("historical evaluate semantic/full binding differs on seal node")
if aggregate_history is not None:
    values = exact_log("logs/aggregate.stdout", ["V4G_CPU_GATE",
        "V4G_AGGREGATE_RUNTIME", "V4G_AGGREGATE_VERIFY", "V4G_CACHE_CLEANED"])
    receipt, digest = historical_receipt("aggregate/receipt.json")
    if (object_sha(values)
            != aggregate_history.get("structured_values_sha256")
            or digest != aggregate_history.get("receipt_digest")
            or receipt.get("status") != aggregate_history.get("receipt_status")
            or receipt.get("metrics", {}).get(
                "exposed_five_view_codec_development_gate")
               is not aggregate_history.get(
                   "exposed_five_view_codec_development_gate")
            or object_sha(receipt.get("controller_barrier_receipt_binding"))
               != aggregate_history.get(
                   "controller_barrier_receipt_binding_sha256")
            or object_sha(receipt.get("inner_receipts"))
               != aggregate_history.get("inner_receipt_binding_ledger_sha256")
            or object_sha(receipt.get("fold_receipts"))
               != aggregate_history.get("fold_receipt_binding_ledger_sha256")
            or object_sha(receipt.get("folds"))
               != aggregate_history.get("fold_value_ledger_sha256")
            or object_sha(receipt.get("global_inner_barrier"))
               != aggregate_history.get("global_inner_barrier_sha256")):
        raise SystemExit("historical aggregate semantic/full binding differs on seal node")
captured_sha = object_sha(captured)
files_sha = object_sha(files)
expected_directories = sorted(actual_directories)
pre_phase_tree_sha = object_sha({"directories": expected_directories, "files": files})
payload = {
    "schema_version": "v4g-controller-phase-state-v1",
    "phase": phase, "reason": reason,
    "allfold_oof_semantic_tensor_read_count_exact0": allfold_oof0,
    "evaluate_fold_command_launch_count": len({
        path for path in evaluate_logs & actual_files if path.endswith(".stdout")}),
    "aggregate_command_launch_count": int("logs/aggregate.stdout" in actual_files),
    "barrier_command_launch_count": int("logs/barrier.stdout" in actual_files),
    "train_command_launch_count": len({path for path in train_logs & actual_files
                                        if path.endswith(".stdout")}),
    "phase_exact_tree_contract": {
        "required_files": sorted(contract["required"]),
        "optional_files_allowed": sorted(contract["optional"]),
        "actual_optional_files": sorted(actual_files - contract["required"]),
        "required_file_count": len(contract["required"]),
        "actual_pre_phase_state_file_count": len(actual_files),
        "required_directories": sorted(contract["required_dirs"]),
        "allowed_directories": sorted(contract["allowed_dirs"]),
        "actual_directory_count": len(actual_directories),
        "no_arbitrary_files_directories_symlinks_or_specials": True,
    },
    "expected_directories": expected_directories,
    "controller_captured_structured_results": structured,
    "controller_captured_structured_results_sha256": object_sha(structured),
    "controller_captured_artifacts": captured,
    "controller_captured_artifacts_sha256": captured_sha,
    "controller_captured_counts": counts,
    "controller_captured_bindings_phase_constraint_satisfied": counts_ok,
    "pre_phase_state_files": files,
    "pre_phase_state_file_count": len(files),
    "pre_phase_state_files_sha256": files_sha,
    "pre_phase_state_tree_sha256": pre_phase_tree_sha,
    "parent_stabilized_inputs": parent_stable,
    "parent_stabilized_inputs_sha256": parent_sha,
    "append_only_historical_ledger": historical,
    "append_only_historical_ledger_sha256": historical_sha,
    "historical_ledger_counts": history_counts,
    "historical_parent_physical_and_seal_stable_binding_reverified": True,
    "historical_semantic_gate_status_full_binding_reverified": True,
    "authority_snapshot": snapshot,
    "source_authority": {
        "release_tree_sha256": release_tree,
        "release_manifest_sha256": manifest_sha,
        "release_manifest_digest": manifest_digest,
        "runtime_sha256": runtime_sha, "runtime_test_sha256": tests_sha,
        "controller_sha256": controller_sha, "python_sha256": python_sha,
    },
    "official_controller_cli_caller_supplied_inner_barrier_or_fold_sha": False,
    "inner_shas_captured_from_verified_child_stdout": counts["train"] == 5,
    "barrier_sha_captured_from_verified_child_stdout": counts["barrier"] == 1,
    "fold_shas_captured_from_verified_child_stdout": counts["evaluate_fold"] == 5,
    "aggregate_sha_captured_from_verified_child_stdout": counts["aggregate"] == 1,
    "inference_authorized": False,
}
payload["phase_state_digest"] = object_sha(payload)
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                 ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
fd = os.open(output, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
             0o400)
with os.fdopen(fd, "w+b") as handle:
    os.fchmod(handle.fileno(), 0o444)
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    written = os.fstat(handle.fileno()); handle.seek(0); readback = handle.read()
    closed = os.fstat(handle.fileno())
after = output.lstat()
if (readback != raw or not (identity(written) == identity(closed) == identity(after))
        or output.is_symlink() or stat.S_IMODE(after.st_mode) != 0o444
        or after.st_nlink != 1):
    raise SystemExit("phase-state same-FD create-only seal differs")
print("V4G_PHASE_STATE=" + json.dumps({
    "path": str(output.resolve(strict=True)),
    "file_sha256": hashlib.sha256(raw).hexdigest(),
    "size_bytes": after.st_size,
    "phase_state_digest": payload["phase_state_digest"], "phase": phase,
    "controller_captured_artifacts_sha256": captured_sha,
    "pre_phase_state_files_sha256": files_sha,
    "pre_phase_state_tree_sha256": pre_phase_tree_sha,
    "append_only_historical_ledger_sha256": historical_sha,
    "allfold_oof_semantic_tensor_read_count_exact0": allfold_oof0,
}, sort_keys=True, separators=(",", ":")))
PY
}

run_seal_child() {
  [[ $# -eq 19 ]] || fail "seal child argument count differs"
  local fn_mode=$1 fn_release_root=$2 fn_python_bin=$3 fn_feature_root=$4
  local fn_v4a_receipt=$5 fn_v4c_receipt=$6 fn_v4d_receipt=$7 fn_run_root=$8
  local fn_job=$9 fn_node=${10} fn_controller_sha=${11} fn_snapshot=${12}
  local fn_phase=${13} fn_allfold_oof0=${14}
  local fn_reason=${15} fn_parent_encoded=${16} fn_parent_sha=${17}
  local fn_history_encoded=${18} fn_history_sha=${19}
  local fn_phase_output fn_phase_transport fn_phase_sha fn_phase_size
  local fn_phase_digest fn_captured_sha fn_prephase_files_sha fn_prephase_tree_sha
  local fn_phase_history_sha
  local fn_controller fn_cache fn_now fn_result
  fn_controller="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_mode}" == success || "${fn_mode}" == no-go ]] || fail "seal mode differs"
  [[ "${fn_job}" == "${cpu_job}" && "${fn_node}" == "${cpu_node}" \
      && "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "seal holder mapping differs"
  require_plain_file "${fn_controller}" "${fn_controller_sha}" 555 seal-controller
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "seal initial authority differs"
  fn_cache="$(fresh_cache "v4g-seal-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1 PYTHONHASHSEED=0
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  cpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    branch-seal v4g-vjepa2-cpu-final-seal-gate-v1 "${cpu_cpus}" >/dev/null
  consumer_wait_stabilized_phase_tree "${fn_python_bin}" "${fn_run_root}" \
    "${fn_parent_encoded}" "${fn_parent_sha}" "${fn_phase}"
  normalize_run_tree_for_phase_state "${fn_run_root}"
  fn_phase_output="$(capture_phase_state "${fn_python_bin}" "${fn_run_root}" \
    "${fn_phase}" "${fn_reason}" "${fn_allfold_oof0}" \
    "${fn_snapshot}" "${fn_controller_sha}" "${fn_run_root}/phase-state.json" \
    "${fn_parent_encoded}" "${fn_parent_sha}" \
    "${fn_history_encoded}" "${fn_history_sha}")"
  [[ "${fn_phase_output}" == V4G_PHASE_STATE=* ]] || \
    fail "seal child phase-state capture differs"
  fn_phase_transport="$(parse_phase_state_output "${fn_python_bin}" \
    "${fn_phase_output}" "${fn_phase}" "${fn_run_root}" "${fn_history_sha}")"
  IFS=: read -r fn_phase_sha fn_phase_size fn_phase_digest fn_captured_sha \
    fn_prephase_files_sha fn_prephase_tree_sha fn_phase_history_sha \
    <<<"${fn_phase_transport}"
  [[ "${fn_phase_history_sha}" == "${fn_history_sha}" ]] || \
    fail "phase historical ledger SHA transport differs"
  find "${fn_run_root}" -type d -exec chmod 0555 {} +
  consumer_wait_file "${fn_python_bin}" "${fn_run_root}/phase-state.json" \
    "${fn_phase_sha}" "${fn_phase_size}" seal-phase-state
  consumer_wait_phase_tree "${fn_python_bin}" "${fn_run_root}/phase-state.json" \
    "${fn_phase_sha}" "${fn_phase_size}"
  fn_result="$("${fn_python_bin}" -I -S -B - \
    "${fn_run_root}" "${fn_mode}" "${fn_phase}" "${fn_allfold_oof0}" \
    "${fn_snapshot}" "${fn_controller_sha}" "${expected_release_tree_sha256}" \
    "${expected_release_manifest_sha256}" "${expected_release_manifest_digest}" \
    "${expected_runtime_sha256}" "${expected_runtime_test_sha256}" \
    "${expected_python_sha256}" "${fn_phase_sha}" "${fn_phase_size}" \
    "${fn_phase_digest}" "${fn_captured_sha}" "${fn_prephase_files_sha}" \
    "${fn_prephase_tree_sha}" "${expected_inner_schema}" \
    "${expected_barrier_schema}" "${expected_fold_schema}" \
    "${expected_aggregate_schema}" "${expected_inner_pass_status}" \
    "${expected_inner_no_go_status}" "${expected_barrier_pass_status}" \
    "${expected_aggregate_status}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

root = Path(sys.argv[1]); mode, phase, allfold_text = sys.argv[2:5]
snapshot, controller_sha, release_tree, manifest_sha, manifest_digest = sys.argv[5:10]
runtime_sha, tests_sha, python_sha = sys.argv[10:13]
phase_sha, phase_size_text, phase_digest, captured_sha = sys.argv[13:17]
prephase_files_sha, prephase_tree_sha = sys.argv[17:19]
inner_schema, barrier_schema, fold_schema, aggregate_schema = sys.argv[19:23]
inner_pass_status, inner_no_go_status = sys.argv[23:25]
barrier_status, aggregate_status = sys.argv[25:27]
if (mode not in {"success", "no-go"} or allfold_text not in {"true", "false"}
        or not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))):
    raise SystemExit("final seal arguments differ")
try:
    phase_size = int(phase_size_text)
except ValueError:
    raise SystemExit("phase-state size argument differs")
if phase_size <= 0:
    raise SystemExit("phase-state size argument differs")
allfold_oof0 = allfold_text == "true"
seal_name = "seal.json" if mode == "success" else "global-no-go-seal.json"
seal_path = root / seal_name
other_seal = root / ("global-no-go-seal.json" if mode == "success" else "seal.json")
if seal_path.exists() or seal_path.is_symlink() or other_seal.exists() or other_seal.is_symlink():
    raise SystemExit("final seal freshness differs")
for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
    if directory.is_symlink(): raise SystemExit("tree directory symlink")
    os.chmod(directory, 0o555)
os.chmod(root, 0o555)
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def object_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()
def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)
def scan(exclude=None):
    members = list(root.rglob("*"))
    if any(path.is_symlink() or (not path.is_dir() and not path.is_file())
           for path in members):
        raise SystemExit("final tree has symlink or special member")
    if stat.S_IMODE(root.lstat().st_mode) != 0o555:
        raise SystemExit("run root mode differs")
    directories = sorted(path.relative_to(root).as_posix()
                         for path in members if path.is_dir())
    if any(stat.S_IMODE((root / relative).lstat().st_mode) != 0o555
           for relative in directories):
        raise SystemExit("final directory mode differs")
    rows = []
    for path in sorted(item for item in members if item.is_file() and item != exclude):
        before = path.lstat()
        if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
                or before.st_nlink != 1 or not hasattr(os, "O_NOFOLLOW")):
            raise SystemExit("final file envelope differs: " + str(path))
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(fd); digest = hashlib.sha256()
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk: break
                digest.update(chunk)
            closed = os.fstat(fd)
        finally:
            os.close(fd)
        after = path.lstat()
        if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
            raise SystemExit("final same-FD identity differs: " + str(path))
        rows.append({"path": path.relative_to(root).as_posix(),
                     "sha256": digest.hexdigest(), "size_bytes": before.st_size,
                     "mode_octal": "0444", "nlink": 1,
                     "physical_identity": {"device": before.st_dev,
                                           "inode": before.st_ino,
                                           "size_bytes": before.st_size,
                                           "mode_octal": format(
                                               stat.S_IMODE(before.st_mode), "04o"),
                                           "nlink": before.st_nlink,
                                           "mtime_ns": before.st_mtime_ns,
                                           "ctime_ns": before.st_ctime_ns}})
    tree = hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()
    return directories, rows, tree

def read_same_fd(relative, expected=None):
    path = root / relative
    if (path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW")
            or str(path) != str(path.resolve(strict=True))):
        raise SystemExit("final semantic path differs: " + relative)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1):
        raise SystemExit("final semantic envelope differs: " + relative)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit("final semantic same-FD identity differs: " + relative)
    binding = {"path": relative, "sha256": digest.hexdigest(),
               "size_bytes": before.st_size, "mode_octal": "0444", "nlink": 1,
               "physical_identity": {"device": before.st_dev, "inode": before.st_ino,
                                     "size_bytes": before.st_size,
                                     "mode_octal": format(
                                         stat.S_IMODE(before.st_mode), "04o"),
                                     "nlink": before.st_nlink,
                                     "mtime_ns": before.st_mtime_ns,
                                     "ctime_ns": before.st_ctime_ns}}
    if expected is not None and any(binding.get(key) != expected.get(key) for key in (
            "path", "sha256", "size_bytes", "mode_octal", "nlink",
            "physical_identity")):
        raise SystemExit("final semantic captured binding differs: " + relative)
    return b"".join(chunks), binding
def read_receipt(relative, expected):
    raw_value, binding = read_same_fd(relative, expected)
    try:
        value = json.loads(raw_value, object_pairs_hook=pairs,
                           parse_constant=nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit("final receipt JSON differs: " + relative + ":" + str(error))
    if type(value) is not dict:
        raise SystemExit("final receipt root differs: " + relative)
    unsigned = dict(value); digest = unsigned.pop("receipt_digest", None)
    if digest != object_sha(unsigned):
        raise SystemExit("final receipt digest differs: " + relative)
    return value, digest, binding

phase_raw, phase_binding = read_same_fd("phase-state.json")
if (phase_binding["sha256"] != phase_sha
        or phase_binding["size_bytes"] != phase_size):
    raise SystemExit("phase-state transported file binding differs")
try:
    phase_value = json.loads(phase_raw, object_pairs_hook=pairs,
                             parse_constant=nonfinite)
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit("phase-state JSON differs: " + str(error))
phase_unsigned = dict(phase_value) if type(phase_value) is dict else {}
actual_phase_digest = phase_unsigned.pop("phase_state_digest", None)
expected_source = {
    "release_tree_sha256": release_tree,
    "release_manifest_sha256": manifest_sha,
    "release_manifest_digest": manifest_digest,
    "runtime_sha256": runtime_sha, "runtime_test_sha256": tests_sha,
    "controller_sha256": controller_sha, "python_sha256": python_sha,
}
parent_stable = phase_value.get("parent_stabilized_inputs")
historical = phase_value.get("append_only_historical_ledger")
historical_sha = phase_value.get("append_only_historical_ledger_sha256")
if (type(phase_value) is not dict
        or actual_phase_digest != object_sha(phase_unsigned)
        or actual_phase_digest != phase_digest
        or phase_value.get("schema_version") != "v4g-controller-phase-state-v1"
        or phase_value.get("phase") != phase
        or phase_value.get("allfold_oof_semantic_tensor_read_count_exact0")
           is not allfold_oof0
        or phase_value.get("authority_snapshot") != snapshot
        or phase_value.get("source_authority") != expected_source
        or phase_value.get("controller_captured_artifacts_sha256") != captured_sha
        or phase_value.get("pre_phase_state_files_sha256") != prephase_files_sha
        or phase_value.get("pre_phase_state_tree_sha256") != prephase_tree_sha
        or type(parent_stable) is not dict
        or object_sha(parent_stable)
           != phase_value.get("parent_stabilized_inputs_sha256")
        or parent_stable.get("phase") != phase
        or type(historical) is not dict or object_sha(historical) != historical_sha
        or historical.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or historical.get("run_root") != str(root)
        or type(historical.get("run_root_physical_identity")) is not dict
        or type(historical.get("run_root_physical_identity", {}).get("device")) is not int
        or type(historical.get("run_root_physical_identity", {}).get("inode")) is not int
        or phase_value.get("historical_ledger_counts")
           != historical.get("captured_counts")
        or phase_value.get(
            "historical_parent_physical_and_seal_stable_binding_reverified") is not True
        or phase_value.get(
            "historical_semantic_gate_status_full_binding_reverified") is not True
        or parent_stable.get("historical_ledger_sha256") != historical_sha
        or parent_stable.get(
            "historical_same_parent_physical_identity_reverified") is not True
        or parent_stable.get(
            "historical_semantic_full_binding_reverified") is not True
        or phase_value.get("official_controller_cli_caller_supplied_inner_barrier_or_fold_sha")
           is not False
        or phase_value.get("inference_authorized") is not False):
    raise SystemExit("phase-state semantic/transport binding differs")
if (mode == "success" and (phase != "SUCCESS" or allfold_oof0
        or not (root / "aggregate" / "receipt.json").is_file())):
    raise SystemExit("success seal branch differs")
if mode == "no-go" and phase == "SUCCESS":
    raise SystemExit("no-go seal branch differs")
if allfold_oof0 and ((root / "aggregate" / "receipt.json").exists()
        or any((root / ("fold" + str(index)) / "fold.json").exists()
               for index in range(5))):
    raise SystemExit("all-fold OOF0 final tree has OOF/aggregate receipt")
directories, preseal_rows, preseal_tree = scan()
phase_row = next((row for row in preseal_rows if row["path"] == "phase-state.json"), None)
if phase_row != phase_binding:
    raise SystemExit("phase-state full binding differs in preseal tree")
phase_files = phase_value.get("pre_phase_state_files")
expected_directories = phase_value.get("expected_directories")
if (type(phase_files) is not list or type(expected_directories) is not list
        or object_sha(phase_files) != prephase_files_sha
        or object_sha({"directories": expected_directories, "files": phase_files})
           != prephase_tree_sha
        or directories != expected_directories
        or preseal_rows != phase_files + [phase_binding]
        or len({row.get("path") for row in phase_files}) != len(phase_files)
        or phase_value.get("pre_phase_state_file_count") != len(phase_files)):
    raise SystemExit("phase-state exact preseal tree binding differs")
if ([{key: row[key] for key in
      ("path", "sha256", "size_bytes", "mode_octal", "nlink")}
     for row in phase_files] != parent_stable.get("files")
        or expected_directories
           != sorted(row.get("path") for row in parent_stable.get("directories", []))):
    raise SystemExit("parent-stable to same-node phase tree join differs")
pre_paths = {row["path"] for row in phase_files}
fold_dirs = {"fold" + str(index) for index in range(5)}
fold_train_files = {
    "fold" + str(index) + "/" + name
    for index in range(5)
    for name in ("preselection.pt", "fixed1200.pt", "inner.json")
}
train_logs = {
    "logs/train-fold" + str(index) + suffix
    for index in range(5) for suffix in (".stdout", ".stderr")
}
barrier_logs = {"logs/barrier.stdout", "logs/barrier.stderr"}
evaluate_logs = {
    "logs/evaluate-fold" + str(index) + suffix
    for index in range(5) for suffix in (".stdout", ".stderr")
}
fold_receipts = {"fold" + str(index) + "/fold.json" for index in range(5)}
aggregate_logs = {"logs/aggregate.stdout", "logs/aggregate.stderr"}
base = {"launch-plan.json"} | train_logs
exact_train = base | fold_train_files
specs = {
    "TRAIN_FAILURE_ALL_OOF0": (base, fold_train_files,
        {"logs"}, {"logs"} | fold_dirs),
    "TRAIN_VERIFICATION_FAILURE_ALL_OOF0": (exact_train, set(),
        {"logs"} | fold_dirs, {"logs"} | fold_dirs),
    "INNER_NO_GO_ALL_OOF0": (exact_train, set(),
        {"logs"} | fold_dirs, {"logs"} | fold_dirs),
    "BARRIER_FAILURE_ALL_OOF0": (exact_train | barrier_logs,
        {"barrier/barrier.json"}, {"logs", "barrier"} | fold_dirs,
        {"logs", "barrier"} | fold_dirs),
    "EVALUATE_FAILURE": (exact_train | barrier_logs | {"barrier/barrier.json"}
        | evaluate_logs, fold_receipts, {"logs", "barrier"} | fold_dirs,
        {"logs", "barrier"} | fold_dirs),
    "AGGREGATE_FAILURE": (exact_train | barrier_logs | {"barrier/barrier.json"}
        | evaluate_logs | fold_receipts | aggregate_logs,
        {"aggregate/receipt.json"},
        {"logs", "barrier", "aggregate"} | fold_dirs,
        {"logs", "barrier", "aggregate"} | fold_dirs),
    "SUCCESS": (exact_train | barrier_logs | {"barrier/barrier.json"}
        | evaluate_logs | fold_receipts | aggregate_logs
        | {"aggregate/receipt.json"}, set(),
        {"logs", "barrier", "aggregate"} | fold_dirs,
        {"logs", "barrier", "aggregate"} | fold_dirs),
}
required, optional, required_dirs, allowed_dirs = specs[phase]
contract = phase_value.get("phase_exact_tree_contract")
if (not required.issubset(pre_paths) or not pre_paths.issubset(required | optional)
        or not required_dirs.issubset(set(directories))
        or not set(directories).issubset(allowed_dirs)
        or type(contract) is not dict
        or contract.get("required_files") != sorted(required)
        or contract.get("optional_files_allowed") != sorted(optional)
        or contract.get("actual_optional_files") != sorted(pre_paths - required)
        or contract.get("required_file_count") != len(required)
        or contract.get("actual_pre_phase_state_file_count") != len(pre_paths)
        or contract.get("required_directories") != sorted(required_dirs)
        or contract.get("allowed_directories") != sorted(allowed_dirs)
        or contract.get("actual_directory_count") != len(directories)
        or contract.get("no_arbitrary_files_directories_symlinks_or_specials") is not True):
    raise SystemExit("final phase-specific exact-tree contract differs")
if phase != "TRAIN_FAILURE_ALL_OOF0" and set(directories) != allowed_dirs:
    raise SystemExit("final phase exact directory set differs")

pre_by_path = {row["path"]: row for row in phase_files}
train_history = historical.get("train_folds")
barrier_history = historical.get("barrier")
evaluate_history = historical.get("evaluate_folds")
aggregate_history = historical.get("aggregate")
history_counts = historical.get("captured_counts")
expected_sequence = (["launch-plan"]
    + ["train-fold" + str(index) for index in range(len(train_history or []))]
    + (["verify-inner-barrier"] if barrier_history is not None else [])
    + ["evaluate-fold" + str(index) for index in range(len(evaluate_history or []))]
    + (["aggregate"] if aggregate_history is not None else []))
if (type(historical.get("launch_plan")) is not dict
        or type(train_history) is not list or type(evaluate_history) is not list
        or historical.get("append_sequence") != expected_sequence
        or [row.get("fold_index") for row in train_history]
           != list(range(len(train_history)))
        or [row.get("fold_index") for row in evaluate_history]
           != list(range(len(evaluate_history)))
        or history_counts != {
            "launch_plan": 1, "train": len(train_history),
            "barrier": int(barrier_history is not None),
            "evaluate_fold": len(evaluate_history),
            "aggregate": int(aggregate_history is not None)}):
    raise SystemExit("final historical ledger sequence/count differs")
history_tuple = (len(train_history), int(barrier_history is not None),
                 len(evaluate_history), int(aggregate_history is not None))
history_phase_ok = {
    "TRAIN_FAILURE_ALL_OOF0": history_tuple == (0, 0, 0, 0),
    "TRAIN_VERIFICATION_FAILURE_ALL_OOF0": (
        0 <= history_tuple[0] < 5 and history_tuple[1:] == (0, 0, 0)),
    "INNER_NO_GO_ALL_OOF0": history_tuple == (5, 0, 0, 0),
    "BARRIER_FAILURE_ALL_OOF0": history_tuple == (5, 0, 0, 0),
    "EVALUATE_FAILURE": (history_tuple[0:2] == (5, 1)
                         and 0 <= history_tuple[2] < 5
                         and history_tuple[3] == 0),
    "AGGREGATE_FAILURE": history_tuple == (5, 1, 5, 0),
    "SUCCESS": history_tuple == (5, 1, 5, 1),
}.get(phase, False)
if not history_phase_ok:
    raise SystemExit("final historical ledger phase count differs")
history_records = ([historical["launch_plan"]] + train_history
                   + ([barrier_history] if barrier_history is not None else [])
                   + evaluate_history
                   + ([aggregate_history] if aggregate_history is not None else []))
historical_stable_by_path = {}
for record in history_records:
    binding_rows = ([record.get("file_binding")]
                    if record is historical["launch_plan"] else record.get("files"))
    if type(binding_rows) is not list:
        raise SystemExit("final historical stage file ledger differs")
    for history_row in binding_rows:
        if type(history_row) is not dict:
            raise SystemExit("final historical file row differs")
        relative = history_row.get("relative_path"); current = pre_by_path.get(relative)
        stable = {"path": relative, "sha256": history_row.get("sha256"),
                  "size_bytes": history_row.get("size_bytes"),
                  "mode_octal": history_row.get("mode_octal"),
                  "nlink": history_row.get("nlink")}
        if (not isinstance(relative, str) or current is None
                or history_row.get("absolute_path") != str(root / relative)
                or history_row.get("same_fd_o_nofollow_verified") is not True
                or stable != {key: current[key] for key in
                               ("path", "sha256", "size_bytes", "mode_octal", "nlink")}
                or relative in historical_stable_by_path
                   and historical_stable_by_path[relative] != stable):
            raise SystemExit("final historical stable file binding differs")
        historical_stable_by_path[relative] = stable
launch_raw, unused_launch_binding = read_same_fd(
    "launch-plan.json", pre_by_path.get("launch-plan.json"))
try:
    launch_value = json.loads(launch_raw, object_pairs_hook=pairs,
                              parse_constant=nonfinite)
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit("final historical launch-plan JSON differs: " + str(error))
if launch_value != historical["launch_plan"].get("value"):
    raise SystemExit("final historical launch-plan full payload differs")
structured = phase_value.get("controller_captured_structured_results")
if (type(structured) is not list
        or object_sha(structured)
           != phase_value.get("controller_captured_structured_results_sha256")):
    raise SystemExit("captured structured log ledger digest differs")
log_cache = {}; seen_log_lines = set()
for row in structured:
    if (type(row) is not dict or type(row.get("line")) is not int
            or row["line"] <= 0 or type(row.get("prefix")) is not str
            or type(row.get("log")) is not str
            or (row["log"], row["line"]) in seen_log_lines):
        raise SystemExit("captured structured log row differs")
    seen_log_lines.add((row["log"], row["line"]))
    if row["log"] not in log_cache:
        raw_log, unused_binding = read_same_fd(row["log"], pre_by_path.get(row["log"]))
        try:
            log_cache[row["log"]] = raw_log.decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as error:
            raise SystemExit("captured structured log UTF-8 differs: " + str(error))
    lines = log_cache[row["log"]]
    if row["line"] > len(lines):
        raise SystemExit("captured structured log line absent")
    line = lines[row["line"] - 1]
    if "=" not in line:
        raise SystemExit("captured structured log line envelope differs")
    prefix, encoded = line.split("=", 1)
    try:
        decoded = json.loads(encoded, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (json.JSONDecodeError, ValueError) as error:
        raise SystemExit("captured structured log JSON differs: " + str(error))
    if prefix != row["prefix"] or decoded != row.get("value"):
        raise SystemExit("captured structured log semantic row differs")

captured = phase_value.get("controller_captured_artifacts")
if (type(captured) is not dict or object_sha(captured) != captured_sha
        or phase_value.get("controller_captured_artifacts_sha256") != captured_sha):
    raise SystemExit("captured artifacts digest differs")
train_captures = captured.get("train_folds")
barrier_capture = captured.get("barrier")
fold_captures = captured.get("evaluate_folds")
aggregate_capture = captured.get("aggregate")
counts = captured.get("counts")
if (type(train_captures) is not list or type(fold_captures) is not list
        or type(counts) is not dict
        or counts != {"train": len(train_captures),
                      "barrier": int(barrier_capture is not None),
                      "evaluate_fold": len(fold_captures),
                      "aggregate": int(aggregate_capture is not None)}
        or phase_value.get("controller_captured_counts") != counts
        or phase_value.get("controller_captured_bindings_phase_constraint_satisfied")
           is not True):
    raise SystemExit("captured artifact count ledger differs")
parent_logical = parent_stable.get("captured_logical")
if (type(parent_logical) is not dict
        or object_sha(parent_logical) != parent_stable.get("captured_logical_sha256")
        or parent_logical.get("counts") != counts):
    raise SystemExit("parent-stable captured logical ledger differs")
if len(structured) != 4 * sum(counts.values()):
    raise SystemExit("captured artifact/structured-log cardinality join differs")
def historical_log_values(relative, prefixes):
    raw_log, unused_binding = read_same_fd(relative, pre_by_path.get(relative))
    try:
        lines = raw_log.decode("utf-8", errors="strict").splitlines()
        if (len(lines) != len(prefixes)
                or [line.split("=", 1)[0] for line in lines] != prefixes):
            raise ValueError("structured prefix/count")
        return [json.loads(line.split("=", 1)[1], object_pairs_hook=pairs,
                           parse_constant=nonfinite) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, IndexError) as error:
        raise SystemExit("final historical log differs: " + relative + ":" + str(error))
for record in train_history:
    index = record["fold_index"]
    values = historical_log_values("logs/train-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME", "V4G_TRAIN_VERIFY",
         "V4G_CACHE_CLEANED"])
    inner, digest, unused_binding = read_receipt(
        "fold" + str(index) + "/inner.json",
        pre_by_path.get("fold" + str(index) + "/inner.json"))
    checkpoint_ledger = {
        "preselection_checkpoint_artifact": inner.get(
            "preselection_checkpoint_artifact"),
        "fixed1200_checkpoint_artifact": inner.get("fixed1200_checkpoint_artifact"),
        "preselection_fixed1200_checkpoint_pair_join": inner.get(
            "preselection_fixed1200_checkpoint_pair_join"),
        "fixed_candidate": inner.get("fixed_candidate"),
    }
    if (object_sha(values) != record.get("structured_values_sha256")
            or digest != record.get("receipt_digest")
            or inner.get("status") != record.get("receipt_status")
            or inner.get("inner_pass") is not record.get("inner_pass")
            or object_sha(checkpoint_ledger)
               != record.get("checkpoint_binding_ledger_sha256")):
        raise SystemExit("final historical train semantic/full binding differs")
if barrier_history is not None:
    values = historical_log_values("logs/barrier.stdout", ["V4G_GPU_GATE",
        "V4G_BARRIER_RUNTIME", "V4G_BARRIER_VERIFY", "V4G_CACHE_CLEANED"])
    receipt, digest, unused_binding = read_receipt(
        "barrier/barrier.json", pre_by_path.get("barrier/barrier.json"))
    if (object_sha(values)
            != barrier_history.get("structured_values_sha256")
            or digest != barrier_history.get("receipt_digest")
            or receipt.get("status") != barrier_history.get("receipt_status")
            or object_sha(receipt.get("members"))
               != barrier_history.get("full_member_binding_ledger_sha256")
            or object_sha(receipt.get("independent_replay_ledger"))
               != barrier_history.get("independent_replay_ledger_sha256")
            or receipt.get("independent_replay_sha256")
               != barrier_history.get("independent_replay_sha256")):
        raise SystemExit("final historical barrier semantic/full binding differs")
for record in evaluate_history:
    index = record["fold_index"]
    values = historical_log_values("logs/evaluate-fold" + str(index) + ".stdout",
        ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME", "V4G_EVALUATE_VERIFY",
         "V4G_CACHE_CLEANED"])
    receipt, digest, unused_binding = read_receipt(
        "fold" + str(index) + "/fold.json",
        pre_by_path.get("fold" + str(index) + "/fold.json"))
    global_barrier = receipt.get("global_inner_barrier")
    global_binding = ({
        "controller_barrier_receipt_binding": global_barrier.get(
            "controller_barrier_receipt_binding"),
        "inner_receipt_bindings": global_barrier.get("inner_receipt_bindings"),
        "independent_replay_ledger_sha256": object_sha(
            global_barrier.get("independent_replay_ledger")),
        "independent_replay_sha256": global_barrier.get(
            "independent_replay_sha256"),
    } if type(global_barrier) is dict else None)
    if (object_sha(values) != record.get("structured_values_sha256")
            or digest != record.get("receipt_digest")
            or receipt.get("status") != record.get("receipt_status")
            or receipt.get("inner_receipt_file_sha256")
               != record.get("inner_receipt_file_sha256")
            or receipt.get("controller_barrier_receipt_file_sha256")
               != record.get("controller_barrier_receipt_file_sha256")
            or object_sha(global_binding)
               != record.get("global_inner_barrier_binding_sha256")
            or object_sha(global_barrier)
               != record.get("global_inner_barrier_sha256")
            or object_sha(receipt.get("fixed1200_checkpoint_artifact"))
               != record.get("fixed1200_checkpoint_artifact_sha256")
            or object_sha(receipt.get("fixed1200_evaluate_checkpoint_binding"))
               != record.get("fixed1200_evaluate_checkpoint_binding_sha256")):
        raise SystemExit("final historical evaluate semantic/full binding differs")
if aggregate_history is not None:
    values = historical_log_values("logs/aggregate.stdout", ["V4G_CPU_GATE",
        "V4G_AGGREGATE_RUNTIME", "V4G_AGGREGATE_VERIFY", "V4G_CACHE_CLEANED"])
    receipt, digest, unused_binding = read_receipt(
        "aggregate/receipt.json", pre_by_path.get("aggregate/receipt.json"))
    if (object_sha(values)
            != aggregate_history.get("structured_values_sha256")
            or digest != aggregate_history.get("receipt_digest")
            or receipt.get("status") != aggregate_history.get("receipt_status")
            or receipt.get("metrics", {}).get(
                "exposed_five_view_codec_development_gate")
               is not aggregate_history.get(
                   "exposed_five_view_codec_development_gate")
            or object_sha(receipt.get("controller_barrier_receipt_binding"))
               != aggregate_history.get(
                   "controller_barrier_receipt_binding_sha256")
            or object_sha(receipt.get("inner_receipts"))
               != aggregate_history.get("inner_receipt_binding_ledger_sha256")
            or object_sha(receipt.get("fold_receipts"))
               != aggregate_history.get("fold_receipt_binding_ledger_sha256")
            or object_sha(receipt.get("folds"))
               != aggregate_history.get("fold_value_ledger_sha256")
            or object_sha(receipt.get("global_inner_barrier"))
               != aggregate_history.get("global_inner_barrier_sha256")):
        raise SystemExit("final historical aggregate semantic/full binding differs")
checkpoint_binding_keys = (
    "path", "file_sha256", "size_bytes", "mode_octal", "nlink",
    "physical_identity", "single_fd_pre_post_sha256_exact",
    "semantic_metadata_state_replay_verified", "checkpoint_role", "outer_fold",
    "metadata_digest", "implementation_sha256", "model_state_sha256",
    "model_fit_original_count", "model_fit_ordered_iids", "model_fit_iid_digest",
    "inner_validation_iid_digest", "fixed_clip_pca_fit_input_sha256",
    "minibatch_schedule_sha256", "runtime_fingerprint",
    "model_schema_reconstructed_and_strict_loaded",
)
def checkpoint_projection(artifact):
    return {key: artifact.get(key) for key in checkpoint_binding_keys}
train_indices = [row.get("fold_index") for row in train_captures]
if train_indices != sorted(set(train_indices)) or any(
        type(index) is not int or not 0 <= index < 5 for index in train_indices):
    raise SystemExit("captured train fold indices differ")
for row in train_captures:
    index = row["fold_index"]
    inner_binding = row.get("inner_receipt")
    pre_binding = row.get("preselection_checkpoint")
    fixed_binding = row.get("fixed1200_checkpoint")
    if not all(type(value) is dict for value in
               (inner_binding, pre_binding, fixed_binding)):
        raise SystemExit("captured train binding shape differs")
    inner, inner_digest, unused_binding = read_receipt(
        "fold" + str(index) + "/inner.json", inner_binding)
    unused_pre, unused_binding = read_same_fd(
        "fold" + str(index) + "/preselection.pt", pre_binding)
    unused_fixed, unused_binding = read_same_fd(
        "fold" + str(index) + "/fixed1200.pt", fixed_binding)
    inner_pass = inner_binding.get("inner_pass")
    expected_status = inner_pass_status if inner_pass is True else (
        inner_no_go_status if inner_pass is False else None)
    pre_artifact = inner.get("preselection_checkpoint_artifact")
    fixed_artifact = inner.get("fixed1200_checkpoint_artifact")
    pair = inner.get("preselection_fixed1200_checkpoint_pair_join")
    expected_inner_runtime = {
        "fold_root": str((root / ("fold" + str(index))).resolve(strict=True)),
        "path": str((root / ("fold" + str(index)) / "inner.json").resolve(
            strict=True)),
        "file_sha256": inner_binding.get("sha256"),
        "receipt_digest": inner_digest, "mode_octal": "0444", "nlink": 1,
        "preselection_checkpoint_metadata_digest": pre_artifact.get(
            "metadata_digest") if type(pre_artifact) is dict else None,
        "fixed1200_checkpoint_metadata_digest": fixed_artifact.get(
            "metadata_digest") if type(fixed_artifact) is dict else None,
        "checkpoint_provenance_binding_sha256": object_sha({
            "preselection": checkpoint_projection(pre_artifact),
            "fixed1200": checkpoint_projection(fixed_artifact),
        }) if type(pre_artifact) is dict and type(fixed_artifact) is dict else None,
    }
    if (inner.get("schema_version") != inner_schema
            or inner.get("status") != expected_status
            or inner_binding.get("status") != expected_status
            or inner.get("fold_index") != index
            or inner.get("inner_pass") is not inner_pass
            or inner_digest != inner_binding.get("receipt_digest")
            or inner.get("oof_semantic_tensor_read_count_exact0") is not True
            or inner.get("oof_semantic_tensor_materialized_count") != 0
            or inner.get("oof_used_for_training_checkpoint_or_inner_gate") is not False
            or type(pre_artifact) is not dict or type(fixed_artifact) is not dict
            or pre_artifact != pre_binding.get("receipt_artifact")
            or fixed_artifact != fixed_binding.get("receipt_artifact")
            or pre_artifact.get("path")
               != str((root / ("fold" + str(index)) / "preselection.pt").resolve(
                   strict=True))
            or pre_artifact.get("file_sha256") != pre_binding.get("sha256")
            or pre_artifact.get("size_bytes") != pre_binding.get("size_bytes")
            or pre_artifact.get("mode_octal") != pre_binding.get("mode_octal")
            or pre_artifact.get("nlink") != pre_binding.get("nlink")
            or pre_artifact.get("physical_identity")
               != pre_binding.get("physical_identity")
            or pre_artifact.get("metadata_digest") != pre_binding.get("metadata_digest")
            or pre_artifact.get("model_state_sha256") != pre_binding.get("model_state_sha256")
            or fixed_artifact.get("path")
               != str((root / ("fold" + str(index)) / "fixed1200.pt").resolve(
                   strict=True))
            or fixed_artifact.get("file_sha256") != fixed_binding.get("sha256")
            or fixed_artifact.get("size_bytes") != fixed_binding.get("size_bytes")
            or fixed_artifact.get("mode_octal") != fixed_binding.get("mode_octal")
            or fixed_artifact.get("nlink") != fixed_binding.get("nlink")
            or fixed_artifact.get("physical_identity")
               != fixed_binding.get("physical_identity")
            or fixed_artifact.get("metadata_digest") != fixed_binding.get("metadata_digest")
            or fixed_artifact.get("model_state_sha256") != fixed_binding.get("model_state_sha256")
            or pair != row.get("preselection_fixed1200_checkpoint_pair_join")
            or fixed_artifact.get("preselection_checkpoint_binding") != {
                key: pre_artifact[key] for key in
                ("path", "file_sha256", "size_bytes", "mode_octal", "nlink",
                 "metadata_digest", "model_state_sha256", "physical_identity")}
            or fixed_artifact.get("preselection_checkpoint_binding_sha256")
               != object_sha(fixed_artifact.get("preselection_checkpoint_binding"))
            or row.get("inner_runtime_binding") != expected_inner_runtime
            or row.get("fixed_candidate_ledger_sha256")
               != object_sha(inner.get("fixed_candidate"))
            or inner.get("qualification_scope", {}).get("inference_authorized") is not False):
        raise SystemExit("final train receipt/checkpoint semantic join differs")

if barrier_capture is not None:
    if type(barrier_capture) is not dict:
        raise SystemExit("captured barrier binding shape differs")
    barrier, barrier_digest, unused_binding = read_receipt(
        "barrier/barrier.json", barrier_capture)
    barrier_members = barrier.get("members")
    barrier_replay = barrier.get("independent_replay_ledger")
    inner_shas = [row["inner_receipt"]["sha256"] for row in train_captures]
    member_shas = ([row.get("inner_receipt_binding", {}).get("file_sha256")
                    for row in barrier_members]
                   if type(barrier_members) is list else None)
    if (barrier.get("schema_version") != barrier_schema
            or barrier.get("status") != barrier_status
            or barrier_capture.get("status") != barrier_status
            or barrier_digest != barrier_capture.get("receipt_digest")
            or type(barrier_members) is not list or len(barrier_members) != 5
            or [row.get("fold_index") for row in barrier_members] != list(range(5))
            or barrier_members != barrier_capture.get("members")
            or barrier_replay != barrier_capture.get("independent_replay_ledger")
            or barrier.get("independent_replay_sha256")
               != barrier_capture.get("independent_replay_sha256")
            or barrier.get("independent_replay_sha256") != object_sha(barrier_replay)
            or member_shas != inner_shas
            or barrier_capture.get("member_inner_receipt_shas") != inner_shas
            or barrier_capture.get("members_sha256") != barrier.get("members_sha256")
            or barrier.get("all_five_exact_one_full_gates_pass") is not True
            or barrier.get("all_five_authority_model_fit_provenances_recomputed") is not True
            or barrier.get("all_five_authority_inner_checkpoint_forwards_reexecuted") is not True
            or barrier.get("oof_semantic_tensor_read_count_exact0") is not True):
        raise SystemExit("final barrier semantic join differs")
    expected_runtime_barrier = {
        "path": str((root / "barrier" / "barrier.json").resolve(strict=True)),
        "file_sha256": barrier_capture["sha256"],
        "receipt_digest": barrier_digest, "mode_octal": "0444", "nlink": 1,
        "controller_expected_sha_exact": True,
    }
    if barrier_capture.get("runtime_barrier_binding") != expected_runtime_barrier:
        raise SystemExit("final barrier runtime binding differs")
    for index, (member, replay_row) in enumerate(zip(barrier_members, barrier_replay)):
        train_row = train_captures[index]
        if (member.get("fold_root") != train_row["inner_runtime_binding"]["fold_root"]
                or member.get("inner_receipt_binding")
                   != train_row["inner_runtime_binding"]
                or member.get("preselection_checkpoint_artifact")
                   != train_row["preselection_checkpoint"]["receipt_artifact"]
                or member.get("fixed1200_checkpoint_artifact")
                   != train_row["fixed1200_checkpoint"]["receipt_artifact"]
                or member.get("preselection_fixed1200_checkpoint_pair_join")
                   != train_row["preselection_fixed1200_checkpoint_pair_join"]
                or member.get("fixed_candidate_ledger_sha256")
                   != train_row["fixed_candidate_ledger_sha256"]
                or member.get("independent_model_fit_provenance_replay_sha256")
                   != replay_row.get("model_fit_provenance_replay_sha256")
                or member.get("independent_inner_replay_sha256")
                   != replay_row.get("inner_replay_sha256")
                or replay_row.get("inner_receipt_file_sha256")
                   != train_row["inner_receipt"]["sha256"]
                or replay_row.get("fixed1200_checkpoint_file_sha256")
                   != train_row["fixed1200_checkpoint"]["sha256"]
                or replay_row.get("fixed1200_model_state_sha256")
                   != train_row["fixed1200_checkpoint"]["model_state_sha256"]
                or replay_row.get("model_fit_provenance_replay_sha256")
                   != object_sha(replay_row.get("model_fit_provenance_replay"))
                or replay_row.get("inner_replay_sha256")
                   != object_sha(replay_row.get("inner_replay_binding"))
                or replay_row.get("preselection_fixed1200_pair_join_sha256")
                   != object_sha(train_row[
                       "preselection_fixed1200_checkpoint_pair_join"])):
            raise SystemExit("final barrier member/provenance full binding differs")
elif any(row.get("barrier_receipt_sha256") for row in fold_captures):
    raise SystemExit("captured folds exist without captured barrier")

fold_indices = [row.get("fold_index") for row in fold_captures]
if fold_indices != sorted(set(fold_indices)) or any(
        type(index) is not int or not 0 <= index < 5 for index in fold_indices):
    raise SystemExit("captured evaluate fold indices differ")
train_by_index = {row["fold_index"]: row for row in train_captures}
for row in fold_captures:
    index = row["fold_index"]
    fold_value, fold_digest, unused_binding = read_receipt(
        "fold" + str(index) + "/fold.json", row)
    inner_sha = train_by_index.get(index, {}).get("inner_receipt", {}).get("sha256")
    global_barrier = fold_value.get("global_inner_barrier")
    expected_train = train_by_index.get(index)
    expected_fixed_artifact = (expected_train.get("fixed1200_checkpoint", {}).get(
        "receipt_artifact") if type(expected_train) is dict else None)
    expected_fixed_binding = (checkpoint_projection(expected_fixed_artifact)
                              if type(expected_fixed_artifact) is dict else None)
    expected_runtime_fold = {
        "fold_root": str((root / ("fold" + str(index))).resolve(strict=True)),
        "path": str((root / ("fold" + str(index)) / "fold.json").resolve(strict=True)),
        "file_sha256": row.get("sha256"), "receipt_digest": fold_digest,
        "mode_octal": "0444", "nlink": 1,
    }
    if (barrier_capture is None or fold_value.get("schema_version") != fold_schema
            or fold_value.get("status") != aggregate_status
            or row.get("status") != aggregate_status
            or fold_value.get("fold_index") != index
            or fold_digest != row.get("receipt_digest")
            or fold_value.get("inner_receipt_file_sha256") != inner_sha
            or row.get("inner_receipt_sha256") != inner_sha
            or fold_value.get("controller_barrier_receipt_file_sha256")
               != barrier_capture.get("sha256")
            or row.get("barrier_receipt_sha256") != barrier_capture.get("sha256")
            or type(global_barrier) is not dict
            or global_barrier.get("controller_barrier_receipt_digest")
               != barrier_capture.get("receipt_digest")
            or global_barrier.get("controller_barrier_receipt_binding")
               != barrier_capture.get("runtime_barrier_binding")
            or global_barrier.get("inner_receipt_bindings")
               != [item["inner_runtime_binding"] for item in train_captures]
            or global_barrier.get("independent_replay_ledger")
               != barrier_capture.get("independent_replay_ledger")
            or global_barrier.get("independent_replay_sha256")
               != barrier_capture.get("independent_replay_sha256")
            or fold_value.get("fixed1200_checkpoint_artifact")
               != expected_fixed_artifact
            or fold_value.get("fixed1200_evaluate_checkpoint_binding")
               != expected_fixed_binding
            or row.get("current_fixed1200_binding") != expected_fixed_binding
            or row.get("runtime_fold_binding") != expected_runtime_fold
            or fold_value.get("oof_used_for_training_checkpoint_inner_gate_or_selection")
               is not False
            or fold_value.get("qualification_scope", {}).get("inference_authorized")
               is not False):
        raise SystemExit("final fold receipt semantic join differs")

aggregate_gate = None
if aggregate_capture is not None:
    if type(aggregate_capture) is not dict or barrier_capture is None:
        raise SystemExit("captured aggregate binding shape differs")
    aggregate_value, aggregate_digest, unused_binding = read_receipt(
        "aggregate/receipt.json", aggregate_capture)
    fold_shas = [row["sha256"] for row in fold_captures]
    runtime_inner_bindings = [row["inner_runtime_binding"] for row in train_captures]
    runtime_fold_bindings = [row["runtime_fold_binding"] for row in fold_captures]
    current_fold_values = [read_receipt(
        "fold" + str(index) + "/fold.json", fold_captures[index])[0]
        for index in range(5)]
    aggregate_gate = aggregate_capture.get(
        "exposed_five_view_codec_development_gate")
    if (aggregate_value.get("schema_version") != aggregate_schema
            or aggregate_value.get("status") != aggregate_status
            or aggregate_capture.get("status") != aggregate_status
            or aggregate_digest != aggregate_capture.get("receipt_digest")
            or aggregate_capture.get("barrier_receipt_sha256")
               != barrier_capture.get("sha256")
            or aggregate_value.get("controller_barrier_receipt_binding")
               != barrier_capture.get("runtime_barrier_binding")
            or aggregate_capture.get("runtime_barrier_binding")
               != barrier_capture.get("runtime_barrier_binding")
            or aggregate_value.get("controller_barrier_receipt_digest")
               != barrier_capture.get("receipt_digest")
            or aggregate_capture.get("fold_receipt_shas") != fold_shas
            or aggregate_value.get("inner_receipts")
               != {"count": 5, "bindings": runtime_inner_bindings}
            or aggregate_capture.get("runtime_inner_receipt_bindings")
               != runtime_inner_bindings
            or aggregate_value.get("fold_receipts")
               != {"count": 5, "bindings": runtime_fold_bindings}
            or aggregate_capture.get("runtime_fold_receipt_bindings")
               != runtime_fold_bindings
            or aggregate_value.get("folds") != current_fold_values
            or aggregate_value.get("global_inner_barrier", {}).get(
                "inner_receipt_bindings") != runtime_inner_bindings
            or aggregate_value.get("global_inner_barrier", {}).get(
                "common_independent_replay_ledger")
               != barrier_capture.get("independent_replay_ledger")
            or aggregate_value.get("global_inner_barrier", {}).get(
                "common_independent_replay_sha256")
               != barrier_capture.get("independent_replay_sha256")
            or type(aggregate_gate) is not bool
            or aggregate_value.get("metrics", {}).get(
                "exposed_five_view_codec_development_gate") is not aggregate_gate
            or aggregate_value.get("qualification_scope", {}).get(
                "aggregate_gate_evaluated") is not True
            or aggregate_value.get("qualification_scope", {}).get(
                "inference_authorized") is not False
            or aggregate_capture.get("inference_authorized") is not False):
        raise SystemExit("final aggregate gate/status semantic join differs")

passes = [row["inner_receipt"]["inner_pass"] for row in train_captures]
if phase in {"TRAIN_FAILURE_ALL_OOF0", "TRAIN_VERIFICATION_FAILURE_ALL_OOF0"}:
    counts_ok = (0 <= counts["train"] <= 5 and counts["barrier"] == 0
                 and counts["evaluate_fold"] == 0 and counts["aggregate"] == 0)
elif phase == "INNER_NO_GO_ALL_OOF0":
    counts_ok = (counts == {"train": 5, "barrier": 0,
                            "evaluate_fold": 0, "aggregate": 0}
                 and any(value is False for value in passes))
elif phase == "BARRIER_FAILURE_ALL_OOF0":
    counts_ok = (counts["train"] == 5 and all(value is True for value in passes)
                 and counts["barrier"] in {0, 1}
                 and counts["evaluate_fold"] == 0 and counts["aggregate"] == 0)
elif phase == "EVALUATE_FAILURE":
    counts_ok = (counts["train"] == 5 and all(value is True for value in passes)
                 and counts["barrier"] == 1
                 and 0 <= counts["evaluate_fold"] <= 5 and counts["aggregate"] == 0)
elif phase == "AGGREGATE_FAILURE":
    counts_ok = (counts["train"] == 5 and all(value is True for value in passes)
                 and counts["barrier"] == 1 and counts["evaluate_fold"] == 5
                 and counts["aggregate"] in {0, 1})
else:
    counts_ok = (counts == {"train": 5, "barrier": 1,
                            "evaluate_fold": 5, "aggregate": 1}
                 and all(value is True for value in passes))
if not counts_ok:
    raise SystemExit("final phase captured artifact state machine differs")
expected_launch_counts = {
    "train": 5,
    "barrier": 0 if phase in {"TRAIN_FAILURE_ALL_OOF0",
                               "TRAIN_VERIFICATION_FAILURE_ALL_OOF0",
                               "INNER_NO_GO_ALL_OOF0"} else 1,
    "evaluate_fold": 5 if phase in {"EVALUATE_FAILURE", "AGGREGATE_FAILURE",
                                     "SUCCESS"} else 0,
    "aggregate": 1 if phase in {"AGGREGATE_FAILURE", "SUCCESS"} else 0,
}
if (phase_value.get("train_command_launch_count") != expected_launch_counts["train"]
        or phase_value.get("barrier_command_launch_count") != expected_launch_counts["barrier"]
        or phase_value.get("evaluate_fold_command_launch_count")
           != expected_launch_counts["evaluate_fold"]
        or phase_value.get("aggregate_command_launch_count")
           != expected_launch_counts["aggregate"]):
    raise SystemExit("final phase command launch count differs")
seal = {
    "schema_version": "v4g-controller-final-run-seal-v1",
    "status": "V4G_EXACT5_SUCCESS_SEALED" if mode == "success"
              else "V4G_GLOBAL_NO_GO_SEALED",
    "branch": mode, "phase": phase,
    "allfold_oof_semantic_tensor_read_count_exact0": allfold_oof0,
    "run_root": str(root), "exact_directories": directories,
    "preseal_exact_file_count": len(preseal_rows),
    "preseal_files": preseal_rows, "preseal_tree_sha256": preseal_tree,
    "phase_state_binding": phase_row,
    "phase_state_digest": actual_phase_digest,
    "phase_pre_state_files_sha256": prephase_files_sha,
    "phase_pre_state_tree_sha256": prephase_tree_sha,
    "parent_stabilized_inputs_sha256": phase_value[
        "parent_stabilized_inputs_sha256"],
    "parent_stabilized_input_tree_sha256": parent_stable["tree_sha256"],
    "append_only_historical_ledger_sha256": historical_sha,
    "append_only_historical_ledger_counts": history_counts,
    "append_only_historical_ledger_sequence": expected_sequence,
    "historical_run_root_parent_physical_identity": historical[
        "run_root_physical_identity"],
    "historical_parent_physical_identity_reverified": True,
    "historical_seal_cross_client_stable_binding_reverified": True,
    "historical_semantic_gate_status_full_binding_reverified": True,
    "controller_captured_artifacts": captured,
    "controller_captured_artifacts_sha256": captured_sha,
    "controller_captured_counts": counts,
    "aggregate_receipt_status": (
        aggregate_capture.get("status") if aggregate_capture is not None else None),
    "aggregate_exposed_five_view_codec_development_gate": aggregate_gate,
    "stage_receipt_checkpoint_log_semantic_join_reverified": True,
    "authority_snapshot": snapshot,
    "source_authority": {
        "release_tree_sha256": release_tree,
        "release_manifest_sha256": manifest_sha,
        "release_manifest_digest": manifest_digest,
        "runtime_sha256": runtime_sha, "runtime_test_sha256": tests_sha,
        "controller_sha256": controller_sha, "python_sha256": python_sha,
    },
    "single_direction_sha256_dag": True,
    "runtime_reverse_pins_controller_or_manifest": False,
    "all_preseal_files_same_fd_o_nofollow_verified": True,
    "all_expected_logs_same_fd_and_structured_rows_reverified": True,
    "all_captured_receipts_and_checkpoints_same_fd_reverified": True,
    "all_directories_mode_0555": True,
    "all_files_mode_0444_nlink1": True,
    "inference_authorized": False,
}
seal["seal_digest"] = hashlib.sha256(json.dumps(
    seal, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).hexdigest()
raw = json.dumps(seal, sort_keys=True, separators=(",", ":"),
                 ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
os.chmod(root, 0o755)
fd = os.open(seal_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
             0o400)
with os.fdopen(fd, "w+b") as handle:
    os.fchmod(handle.fileno(), 0o444)
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    written = os.fstat(handle.fileno()); handle.seek(0); readback = handle.read()
    closed = os.fstat(handle.fileno())
after = seal_path.lstat(); os.chmod(root, 0o555); root_after = root.lstat()
if (readback != raw or not (identity(written) == identity(closed) == identity(after))
        or seal_path.is_symlink() or stat.S_IMODE(after.st_mode) != 0o444
        or after.st_nlink != 1 or stat.S_IMODE(root_after.st_mode) != 0o555):
    raise SystemExit("final seal same-FD create-only closure differs")
final_directories, final_rows, final_tree = scan()
final_by_path = {row["path"]: row for row in final_rows}
final_stable_files = [{key: row[key] for key in
                       ("path", "sha256", "size_bytes", "mode_octal", "nlink")}
                      for row in final_rows]
final_stable_tree = object_sha({"directories": final_directories,
                                "files": final_stable_files})
if (final_directories != directories
        or set(final_by_path) != {row["path"] for row in preseal_rows} | {seal_name}
        or len(final_rows) != len(preseal_rows) + 1
        or any(final_by_path.get(row["path"]) != row for row in preseal_rows)
        or final_by_path.get(seal_name, {}).get("sha256")
           != hashlib.sha256(raw).hexdigest()):
    raise SystemExit("final exact-tree closure differs")
print(json.dumps({
    "seal": str(seal_path.resolve(strict=True)),
    "seal_sha256": hashlib.sha256(raw).hexdigest(),
    "seal_digest": seal["seal_digest"], "branch": mode, "phase": phase,
    "preseal_exact_file_count": len(preseal_rows),
    "final_exact_file_count": len(final_rows), "final_tree_sha256": final_tree,
    "final_stable_tree_sha256": final_stable_tree,
    "append_only_historical_ledger_sha256": historical_sha,
    "allfold_oof_semantic_tensor_read_count_exact0": allfold_oof0,
}, sort_keys=True, separators=(",", ":")))
PY
  )"
  fn_now="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
    "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_receipt}" \
    "${fn_v4d_receipt}" "${fn_controller}" "${fn_controller_sha}")"
  [[ "${fn_now}" == "${fn_snapshot}" ]] || fail "seal changed authority"
  trap - EXIT; cleanup_cache "${fn_cache}"
  printf '%s\n' "${fn_phase_output}"
  if [[ "${fn_mode}" == success ]]; then
    printf 'V4G_FINAL_SEAL=%s\n' "${fn_result}"
  else
    printf 'V4G_GLOBAL_NO_GO_SEAL=%s\n' "${fn_result}"
  fi
}

normalize_run_tree_for_phase_state() {
  local fn_root=$1 fn_special
  [[ "${fn_root}" == /* && -d "${fn_root}" && ! -L "${fn_root}" ]] || \
    fail "phase normalization root differs"
  fn_special="$(find "${fn_root}" -mindepth 1 ! -type f ! -type d -print -quit)"
  [[ -z "${fn_special}" ]] || fail "phase tree has symlink/special member"
  find "${fn_root}" -type f -exec chmod 0444 {} +
  find "${fn_root}" -type d -exec chmod 0700 {} +
}

parse_train_log() {
  [[ $# -eq 6 ]] || fail "parse_train_log argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_fold=$3 fn_run_root=$4
  local fn_history_encoded=$5 fn_history_sha=$6
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_fold}" \
    "${fn_run_root}" "${fn_history_encoded}" "${fn_history_sha}" <<'PY'
from pathlib import Path
import base64, binascii, hashlib, json, os, stat, sys, time
path = Path(sys.argv[1]); fold = int(sys.argv[2]); root = Path(sys.argv[3])
encoded, expected_ledger_sha = sys.argv[4:6]
def pairs(values):
    result = {}
    for key, value in values:
        if key in result: raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value): raise ValueError("nonfinite JSON constant: " + value)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
try:
    ledger_raw = base64.b64decode(encoded.encode("ascii"), altchars=b"-_",
                                  validate=True)
    ledger = json.loads(ledger_raw, object_pairs_hook=pairs,
                        parse_constant=nonfinite)
except (ValueError, binascii.Error, json.JSONDecodeError) as error:
    raise SystemExit("train historical ledger decode differs: " + str(error))
expected_sequence = ["launch-plan"] + [
    "train-fold" + str(index) for index in range(len(ledger.get("train_folds", [])))]
if (len(encoded) > 98304
        or canonical(ledger) != ledger_raw
        or hashlib.sha256(ledger_raw).hexdigest() != expected_ledger_sha
        or base64.b64encode(ledger_raw, altchars=b"-_").decode("ascii") != encoded
        or ledger.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or ledger.get("run_root") != str(root.resolve(strict=True))
        or ledger.get("run_root_physical_identity") != {
            "device": root.lstat().st_dev, "inode": root.lstat().st_ino}
        or ledger.get("append_sequence") != expected_sequence
        or fold != len(ledger.get("train_folds", []))
        or ledger.get("barrier") is not None
        or ledger.get("evaluate_folds") != [] or ledger.get("aggregate") is not None
        or ledger.get("captured_counts") != {
            "launch_plan": 1, "train": fold, "barrier": 0,
            "evaluate_fold": 0, "aggregate": 0}):
    raise SystemExit("train historical ledger prefix/append order differs")
identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def bind_once(relative):
    target = root / relative; before = target.lstat()
    if (target.is_symlink() or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or str(target) != str(target.resolve(strict=True))
            or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("file envelope differs: " + relative)
    fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = target.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise RuntimeError("same-FD identity differs: " + relative)
    return b"".join(chunks), {
        "relative_path": relative, "absolute_path": str(target.resolve(strict=True)),
        "sha256": digest.hexdigest(), "size_bytes": before.st_size,
        "mode_octal": "0444", "nlink": 1,
        "physical_identity": {"device": before.st_dev, "inode": before.st_ino,
                              "size_bytes": before.st_size,
                              "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
                              "nlink": before.st_nlink,
                              "mtime_ns": before.st_mtime_ns,
                              "ctime_ns": before.st_ctime_ns},
        "same_fd_o_nofollow_verified": True,
    }
def bind_wait(relative):
    last = "not attempted"
    for attempt in range(1, 21):
        try:
            return bind_once(relative)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last = type(error).__name__ + ":" + str(error)
            print("V4G_HISTORY_NFS_RETRY stage=train fold=" + str(fold)
                  + " path=" + relative + " attempt=" + str(attempt)
                  + " diagnostic=" + last, file=sys.stderr, flush=True)
            if attempt != 20: time.sleep(1.0)
    raise SystemExit("train historical capture timeout: " + last)
expected_relative = "logs/train-fold" + str(fold) + ".stdout"
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or path != root / expected_relative):
    raise SystemExit("train historical path differs")
relative_paths = [expected_relative, "logs/train-fold" + str(fold) + ".stderr",
                  "fold" + str(fold) + "/inner.json",
                  "fold" + str(fold) + "/preselection.pt",
                  "fold" + str(fold) + "/fixed1200.pt"]
captures = [bind_wait(relative) for relative in relative_paths]
raws = {relative: captured[0] for relative, captured in zip(relative_paths, captures)}
bindings = [captured[1] for captured in captures]
try:
    lines = raws[expected_relative].decode("utf-8", errors="strict").splitlines()
except UnicodeDecodeError as error:
    raise SystemExit("train structured stdout UTF-8 differs: " + str(error))
prefixes = ["V4G_GPU_GATE", "V4G_TRAIN_RUNTIME", "V4G_TRAIN_VERIFY",
            "V4G_CACHE_CLEANED"]
if len(lines) != 4 or [line.split("=", 1)[0] for line in lines] != prefixes:
    raise SystemExit("train structured stdout exact4 differs")
try:
    values = [json.loads(line.split("=", 1)[1], object_pairs_hook=pairs,
                         parse_constant=nonfinite) for line in lines]
    inner = json.loads(raws[relative_paths[2]], object_pairs_hook=pairs,
                       parse_constant=nonfinite)
except (json.JSONDecodeError, ValueError) as error:
    raise SystemExit("train historical structured JSON differs: " + str(error))
runtime, verify = values[1], values[2]
unsigned = dict(inner); inner_digest = unsigned.pop("receipt_digest", None)
by_relative = {row["relative_path"]: row for row in bindings}
inner_binding = by_relative[relative_paths[2]]
pre_binding = by_relative[relative_paths[3]]
fixed_binding = by_relative[relative_paths[4]]
if (values[0].get("role") != "train-fold" + str(fold)
        or runtime.get("fold_index") != fold or verify.get("fold_index") != fold
        or runtime.get("inner_receipt_sha256") != verify.get("inner_receipt_sha256")
        or runtime.get("inner_receipt_digest") != verify.get("inner_receipt_digest")
        or runtime.get("inner_pass") is not verify.get("inner_pass")
        or verify.get("oof_semantic_tensor_read_count") != 0
        or values[3].get("role") != "train-fold" + str(fold)
        or values[3].get("absent_after_cleanup") is not True
        or inner.get("schema_version")
           != "semantic-anchor-vjepa2-role-directed-teacher-margin-inner-receipt-v4g"
        or inner_digest != object_sha(unsigned)
        or inner_digest != verify.get("inner_receipt_digest")
        or inner.get("status") != verify.get("inner_status")
        or inner.get("inner_pass") is not verify.get("inner_pass")
        or inner_binding["sha256"] != verify.get("inner_receipt_sha256")
        or inner_binding["size_bytes"] != verify.get("inner_receipt_size_bytes")
        or pre_binding["sha256"] != verify.get("preselection_sha256")
        or pre_binding["size_bytes"] != verify.get("preselection_size_bytes")
        or fixed_binding["sha256"] != verify.get("fixed1200_sha256")
        or fixed_binding["size_bytes"] != verify.get("fixed1200_size_bytes")):
    raise SystemExit("train structured stdout semantic join differs")
record = {
    "stage": "train-fold", "fold_index": fold, "files": bindings,
    "structured_values_sha256": object_sha(values),
    "receipt_digest": inner_digest,
    "receipt_status": inner.get("status"), "inner_pass": inner.get("inner_pass"),
    "checkpoint_binding_ledger_sha256": object_sha({
        "preselection_checkpoint_artifact": inner.get(
            "preselection_checkpoint_artifact"),
        "fixed1200_checkpoint_artifact": inner.get("fixed1200_checkpoint_artifact"),
        "preselection_fixed1200_checkpoint_pair_join": inner.get(
            "preselection_fixed1200_checkpoint_pair_join"),
        "fixed_candidate": inner.get("fixed_candidate"),
    }),
}
ledger["train_folds"].append(record)
ledger["append_sequence"].append("train-fold" + str(fold))
ledger["captured_counts"]["train"] += 1
new_encoded = base64.urlsafe_b64encode(canonical(ledger)).decode("ascii")
if len(new_encoded) > 98304:
    raise SystemExit("train compact historical ledger exceeds argv safety bound")
print(verify["inner_receipt_sha256"] + ":" + verify["inner_receipt_digest"]
      + ":" + str(verify["inner_receipt_size_bytes"])
      + ":" + ("true" if verify["inner_pass"] else "false")
      + ":" + verify["preselection_sha256"]
      + ":" + str(verify["preselection_size_bytes"])
      + ":" + verify["fixed1200_sha256"]
      + ":" + str(verify["fixed1200_size_bytes"])
      + ":" + new_encoded + ":" + object_sha(ledger))
PY
}

parse_barrier_log() {
  [[ $# -eq 5 ]] || fail "parse_barrier_log argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_run_root=$3
  local fn_history_encoded=$4 fn_history_sha=$5
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_run_root}" \
    "${fn_history_encoded}" "${fn_history_sha}" <<'PY'
from pathlib import Path
import base64, binascii, hashlib, json, os, stat, sys, time
path = Path(sys.argv[1]); root = Path(sys.argv[2])
encoded, expected_ledger_sha = sys.argv[3:5]
def pairs(values):
    result = {}
    for key, value in values:
        if key in result: raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value): raise ValueError("nonfinite JSON constant: " + value)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
try:
    ledger_raw = base64.b64decode(encoded.encode("ascii"), altchars=b"-_",
                                  validate=True)
    ledger = json.loads(ledger_raw, object_pairs_hook=pairs,
                        parse_constant=nonfinite)
except (ValueError, binascii.Error, json.JSONDecodeError) as error:
    raise SystemExit("barrier historical ledger decode differs: " + str(error))
expected_sequence = ["launch-plan"] + ["train-fold" + str(i) for i in range(5)]
if (len(encoded) > 98304
        or canonical(ledger) != ledger_raw
        or hashlib.sha256(ledger_raw).hexdigest() != expected_ledger_sha
        or base64.b64encode(ledger_raw, altchars=b"-_").decode("ascii") != encoded
        or ledger.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or ledger.get("run_root") != str(root.resolve(strict=True))
        or ledger.get("run_root_physical_identity") != {
            "device": root.lstat().st_dev, "inode": root.lstat().st_ino}
        or ledger.get("append_sequence") != expected_sequence
        or len(ledger.get("train_folds", [])) != 5
        or ledger.get("barrier") is not None
        or ledger.get("evaluate_folds") != [] or ledger.get("aggregate") is not None
        or ledger.get("captured_counts") != {
            "launch_plan": 1, "train": 5, "barrier": 0,
            "evaluate_fold": 0, "aggregate": 0}):
    raise SystemExit("barrier historical ledger prefix/append order differs")
identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                          value.st_mode, value.st_nlink, value.st_mtime_ns,
                          value.st_ctime_ns)
def bind_once(relative):
    target = root / relative; before = target.lstat()
    if (target.is_symlink() or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or str(target) != str(target.resolve(strict=True))
            or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("file envelope differs: " + relative)
    fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = target.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise RuntimeError("same-FD identity differs: " + relative)
    return b"".join(chunks), {
        "relative_path": relative, "absolute_path": str(target.resolve(strict=True)),
        "sha256": digest.hexdigest(), "size_bytes": before.st_size,
        "mode_octal": "0444", "nlink": 1,
        "physical_identity": {"device": before.st_dev, "inode": before.st_ino,
                              "size_bytes": before.st_size,
                              "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
                              "nlink": before.st_nlink,
                              "mtime_ns": before.st_mtime_ns,
                              "ctime_ns": before.st_ctime_ns},
        "same_fd_o_nofollow_verified": True,
    }
def bind_wait(relative):
    last = "not attempted"
    for attempt in range(1, 21):
        try: return bind_once(relative)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last = type(error).__name__ + ":" + str(error)
            print("V4G_HISTORY_NFS_RETRY stage=barrier path=" + relative
                  + " attempt=" + str(attempt) + " diagnostic=" + last,
                  file=sys.stderr, flush=True)
            if attempt != 20: time.sleep(1.0)
    raise SystemExit("barrier historical capture timeout: " + last)
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or path != root / "logs/barrier.stdout"):
    raise SystemExit("barrier historical path differs")
relative_paths = ["logs/barrier.stdout", "logs/barrier.stderr",
                  "barrier/barrier.json"]
captures = [bind_wait(relative) for relative in relative_paths]
raws = {relative: captured[0] for relative, captured in zip(relative_paths, captures)}
bindings = [captured[1] for captured in captures]
try:
    lines = raws[relative_paths[0]].decode("utf-8", errors="strict").splitlines()
except UnicodeDecodeError as error:
    raise SystemExit("barrier structured stdout UTF-8 differs: " + str(error))
prefixes = ["V4G_GPU_GATE", "V4G_BARRIER_RUNTIME", "V4G_BARRIER_VERIFY",
            "V4G_CACHE_CLEANED"]
if len(lines) != 4 or [line.split("=", 1)[0] for line in lines] != prefixes:
    raise SystemExit("barrier structured stdout exact4 differs")
try:
    values = [json.loads(line.split("=", 1)[1], object_pairs_hook=pairs,
                         parse_constant=nonfinite) for line in lines]
    receipt = json.loads(raws[relative_paths[2]], object_pairs_hook=pairs,
                         parse_constant=nonfinite)
except (json.JSONDecodeError, ValueError) as error:
    raise SystemExit("barrier historical structured JSON differs: " + str(error))
runtime, verify = values[1], values[2]
unsigned = dict(receipt); receipt_digest = unsigned.pop("receipt_digest", None)
receipt_binding = bindings[2]
if (values[0].get("role") != "verify-inner-barrier"
        or runtime.get("barrier_receipt_sha256") != verify.get("barrier_receipt_sha256")
        or runtime.get("barrier_receipt_digest") != verify.get("barrier_receipt_digest")
        or runtime.get("oof_semantic_tensor_read_count") != 0
        or verify.get("oof_semantic_tensor_read_count") != 0
        or values[3].get("role") != "verify-inner-barrier"
        or values[3].get("absent_after_cleanup") is not True
        or receipt.get("schema_version")
           != "semantic-anchor-vjepa2-role-directed-teacher-margin-global-inner-barrier-v4g"
        or receipt.get("status") != "V4G_EXACT5_INNER_BARRIER_PASS_OOF_UNREAD"
        or receipt_digest != object_sha(unsigned)
        or receipt_digest != verify.get("barrier_receipt_digest")
        or receipt_binding["sha256"] != verify.get("barrier_receipt_sha256")
        or receipt_binding["size_bytes"] != verify.get("barrier_receipt_size_bytes")
        or type(receipt.get("members")) is not list
        or len(receipt.get("members")) != 5
        or type(receipt.get("independent_replay_ledger")) is not list):
    raise SystemExit("barrier structured stdout semantic join differs")
record = {
    "stage": "verify-inner-barrier", "files": bindings,
    "structured_values_sha256": object_sha(values),
    "receipt_digest": receipt_digest,
    "receipt_status": receipt.get("status"),
    "full_member_binding_ledger_sha256": object_sha(receipt.get("members")),
    "independent_replay_ledger_sha256": object_sha(
        receipt.get("independent_replay_ledger")),
    "independent_replay_sha256": receipt.get("independent_replay_sha256"),
}
ledger["barrier"] = record
ledger["append_sequence"].append("verify-inner-barrier")
ledger["captured_counts"]["barrier"] = 1
new_encoded = base64.urlsafe_b64encode(canonical(ledger)).decode("ascii")
if len(new_encoded) > 98304:
    raise SystemExit("barrier compact historical ledger exceeds argv safety bound")
print(verify["barrier_receipt_sha256"] + ":" + verify["barrier_receipt_digest"]
      + ":" + str(verify["barrier_receipt_size_bytes"])
      + ":" + new_encoded + ":" + object_sha(ledger))
PY
}

parse_evaluate_log() {
  [[ $# -eq 8 ]] || fail "parse_evaluate_log argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_fold=$3 fn_barrier_sha=$4 fn_inner_sha=$5
  local fn_run_root=$6 fn_history_encoded=$7 fn_history_sha=$8
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_fold}" \
    "${fn_barrier_sha}" "${fn_inner_sha}" "${fn_run_root}" \
    "${fn_history_encoded}" "${fn_history_sha}" <<'PY'
from pathlib import Path
import base64, binascii, hashlib, json, os, stat, sys, time
path = Path(sys.argv[1]); fold = int(sys.argv[2]); expected_barrier = sys.argv[3]
expected_inner = sys.argv[4]; root = Path(sys.argv[5])
encoded, expected_ledger_sha = sys.argv[6:8]
def pairs(values):
    result = {}
    for key, value in values:
        if key in result: raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value): raise ValueError("nonfinite JSON constant: " + value)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
try:
    ledger_raw = base64.b64decode(encoded.encode("ascii"), altchars=b"-_",
                                  validate=True)
    ledger = json.loads(ledger_raw, object_pairs_hook=pairs,
                        parse_constant=nonfinite)
except (ValueError, binascii.Error, json.JSONDecodeError) as error:
    raise SystemExit("evaluate historical ledger decode differs: " + str(error))
expected_sequence = (["launch-plan"] + ["train-fold" + str(i) for i in range(5)]
                     + ["verify-inner-barrier"]
                     + ["evaluate-fold" + str(i)
                        for i in range(len(ledger.get("evaluate_folds", [])))])
if (len(encoded) > 98304
        or canonical(ledger) != ledger_raw
        or hashlib.sha256(ledger_raw).hexdigest() != expected_ledger_sha
        or base64.b64encode(ledger_raw, altchars=b"-_").decode("ascii") != encoded
        or ledger.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or ledger.get("run_root") != str(root.resolve(strict=True))
        or ledger.get("run_root_physical_identity") != {
            "device": root.lstat().st_dev, "inode": root.lstat().st_ino}
        or ledger.get("append_sequence") != expected_sequence
        or len(ledger.get("train_folds", [])) != 5
        or type(ledger.get("barrier")) is not dict
        or fold != len(ledger.get("evaluate_folds", []))
        or ledger.get("aggregate") is not None
        or ledger.get("captured_counts") != {
            "launch_plan": 1, "train": 5, "barrier": 1,
            "evaluate_fold": fold, "aggregate": 0}):
    raise SystemExit("evaluate historical ledger prefix/append order differs")
identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                          value.st_mode, value.st_nlink, value.st_mtime_ns,
                          value.st_ctime_ns)
def bind_once(relative):
    target = root / relative; before = target.lstat()
    if (target.is_symlink() or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or str(target) != str(target.resolve(strict=True))
            or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("file envelope differs: " + relative)
    fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = target.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise RuntimeError("same-FD identity differs: " + relative)
    return b"".join(chunks), {
        "relative_path": relative, "absolute_path": str(target.resolve(strict=True)),
        "sha256": digest.hexdigest(), "size_bytes": before.st_size,
        "mode_octal": "0444", "nlink": 1,
        "physical_identity": {"device": before.st_dev, "inode": before.st_ino,
                              "size_bytes": before.st_size,
                              "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
                              "nlink": before.st_nlink,
                              "mtime_ns": before.st_mtime_ns,
                              "ctime_ns": before.st_ctime_ns},
        "same_fd_o_nofollow_verified": True,
    }
def bind_wait(relative):
    last = "not attempted"
    for attempt in range(1, 21):
        try: return bind_once(relative)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last = type(error).__name__ + ":" + str(error)
            print("V4G_HISTORY_NFS_RETRY stage=evaluate fold=" + str(fold)
                  + " path=" + relative + " attempt=" + str(attempt)
                  + " diagnostic=" + last, file=sys.stderr, flush=True)
            if attempt != 20: time.sleep(1.0)
    raise SystemExit("evaluate historical capture timeout: " + last)
stdout_relative = "logs/evaluate-fold" + str(fold) + ".stdout"
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or path != root / stdout_relative):
    raise SystemExit("evaluate historical path differs")
relative_paths = [stdout_relative, "logs/evaluate-fold" + str(fold) + ".stderr",
                  "fold" + str(fold) + "/fold.json", "barrier/barrier.json",
                  "fold" + str(fold) + "/fixed1200.pt"]
captures = [bind_wait(relative) for relative in relative_paths]
raws = {relative: captured[0] for relative, captured in zip(relative_paths, captures)}
bindings = [captured[1] for captured in captures]
try:
    lines = raws[stdout_relative].decode("utf-8", errors="strict").splitlines()
except UnicodeDecodeError as error:
    raise SystemExit("evaluate structured stdout UTF-8 differs: " + str(error))
prefixes = ["V4G_GPU_GATE", "V4G_EVALUATE_RUNTIME", "V4G_EVALUATE_VERIFY",
            "V4G_CACHE_CLEANED"]
if len(lines) != 4 or [line.split("=", 1)[0] for line in lines] != prefixes:
    raise SystemExit("evaluate structured stdout exact4 differs")
try:
    values = [json.loads(line.split("=", 1)[1], object_pairs_hook=pairs,
                         parse_constant=nonfinite) for line in lines]
    receipt = json.loads(raws[relative_paths[2]], object_pairs_hook=pairs,
                         parse_constant=nonfinite)
except (json.JSONDecodeError, ValueError) as error:
    raise SystemExit("evaluate historical structured JSON differs: " + str(error))
runtime, verify = values[1], values[2]
unsigned = dict(receipt); receipt_digest = unsigned.pop("receipt_digest", None)
by_relative = {row["relative_path"]: row for row in bindings}
receipt_binding = by_relative[relative_paths[2]]
barrier_binding = by_relative[relative_paths[3]]
fixed_binding = by_relative[relative_paths[4]]
global_barrier = receipt.get("global_inner_barrier")
fixed_artifact = receipt.get("fixed1200_checkpoint_artifact")
if (values[0].get("role") != "evaluate-fold" + str(fold)
        or runtime.get("fold_index") != fold or verify.get("fold_index") != fold
        or runtime.get("fold_receipt_sha256") != verify.get("fold_receipt_sha256")
        or runtime.get("fold_receipt_digest") != verify.get("fold_receipt_digest")
        or verify.get("barrier_receipt_sha256") != expected_barrier
        or verify.get("inner_receipt_sha256") != expected_inner
        or values[3].get("role") != "evaluate-fold" + str(fold)
        or values[3].get("absent_after_cleanup") is not True
        or receipt.get("schema_version")
           != "semantic-anchor-vjepa2-role-directed-teacher-margin-fold-receipt-v4g"
        or receipt.get("status")
           != "V4G_ROLE_DIRECTED_TEACHER_MARGIN_KNOWN_EXPOSED_DEVELOPMENT"
        or receipt_digest != object_sha(unsigned)
        or receipt_digest != verify.get("fold_receipt_digest")
        or receipt_binding["sha256"] != verify.get("fold_receipt_sha256")
        or receipt_binding["size_bytes"] != verify.get("fold_receipt_size_bytes")
        or barrier_binding["sha256"] != expected_barrier
        or receipt.get("controller_barrier_receipt_file_sha256") != expected_barrier
        or receipt.get("inner_receipt_file_sha256") != expected_inner
        or type(global_barrier) is not dict or type(fixed_artifact) is not dict
        or fixed_artifact.get("file_sha256") != fixed_binding["sha256"]
        or fixed_artifact.get("size_bytes") != fixed_binding["size_bytes"]):
    raise SystemExit("evaluate structured stdout semantic join differs")
record = {
    "stage": "evaluate-fold", "fold_index": fold, "files": bindings,
    "structured_values_sha256": object_sha(values),
    "receipt_digest": receipt_digest,
    "receipt_status": receipt.get("status"),
    "inner_receipt_file_sha256": receipt.get("inner_receipt_file_sha256"),
    "controller_barrier_receipt_file_sha256": receipt.get(
        "controller_barrier_receipt_file_sha256"),
    "global_inner_barrier_binding_sha256": object_sha({
        "controller_barrier_receipt_binding": global_barrier.get(
            "controller_barrier_receipt_binding"),
        "inner_receipt_bindings": global_barrier.get("inner_receipt_bindings"),
        "independent_replay_ledger_sha256": object_sha(
            global_barrier.get("independent_replay_ledger")),
        "independent_replay_sha256": global_barrier.get(
            "independent_replay_sha256"),
    }),
    "global_inner_barrier_sha256": object_sha(global_barrier),
    "fixed1200_checkpoint_artifact_sha256": object_sha(fixed_artifact),
    "fixed1200_evaluate_checkpoint_binding_sha256": object_sha(
        receipt.get("fixed1200_evaluate_checkpoint_binding")),
}
ledger["evaluate_folds"].append(record)
ledger["append_sequence"].append("evaluate-fold" + str(fold))
ledger["captured_counts"]["evaluate_fold"] += 1
new_encoded = base64.urlsafe_b64encode(canonical(ledger)).decode("ascii")
if len(new_encoded) > 98304:
    raise SystemExit("evaluate compact historical ledger exceeds argv safety bound")
print(verify["fold_receipt_sha256"] + ":" + verify["fold_receipt_digest"]
      + ":" + str(verify["fold_receipt_size_bytes"])
      + ":" + new_encoded + ":" + object_sha(ledger))
PY
}

parse_aggregate_log() {
  [[ $# -eq 6 ]] || fail "parse_aggregate_log argument count differs"
  local fn_python_bin=$1 fn_path=$2 fn_barrier_sha=$3 fn_run_root=$4
  local fn_history_encoded=$5 fn_history_sha=$6
  "${fn_python_bin}" -I -S -B - "${fn_path}" "${fn_barrier_sha}" \
    "${fn_run_root}" "${fn_history_encoded}" "${fn_history_sha}" <<'PY'
from pathlib import Path
import base64, binascii, hashlib, json, os, stat, sys, time
path = Path(sys.argv[1]); expected_barrier = sys.argv[2]; root = Path(sys.argv[3])
encoded, expected_ledger_sha = sys.argv[4:6]
def pairs(values):
    result = {}
    for key, value in values:
        if key in result: raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value): raise ValueError("nonfinite JSON constant: " + value)
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def object_sha(value): return hashlib.sha256(canonical(value)).hexdigest()
try:
    ledger_raw = base64.b64decode(encoded.encode("ascii"), altchars=b"-_",
                                  validate=True)
    ledger = json.loads(ledger_raw, object_pairs_hook=pairs,
                        parse_constant=nonfinite)
except (ValueError, binascii.Error, json.JSONDecodeError) as error:
    raise SystemExit("aggregate historical ledger decode differs: " + str(error))
expected_sequence = (["launch-plan"] + ["train-fold" + str(i) for i in range(5)]
                     + ["verify-inner-barrier"]
                     + ["evaluate-fold" + str(i) for i in range(5)])
if (len(encoded) > 98304
        or canonical(ledger) != ledger_raw
        or hashlib.sha256(ledger_raw).hexdigest() != expected_ledger_sha
        or base64.b64encode(ledger_raw, altchars=b"-_").decode("ascii") != encoded
        or ledger.get("schema_version")
           != "v4g-controller-append-only-historical-ledger-v1"
        or ledger.get("run_root") != str(root.resolve(strict=True))
        or ledger.get("run_root_physical_identity") != {
            "device": root.lstat().st_dev, "inode": root.lstat().st_ino}
        or ledger.get("append_sequence") != expected_sequence
        or len(ledger.get("train_folds", [])) != 5
        or type(ledger.get("barrier")) is not dict
        or len(ledger.get("evaluate_folds", [])) != 5
        or ledger.get("aggregate") is not None
        or ledger.get("captured_counts") != {
            "launch_plan": 1, "train": 5, "barrier": 1,
            "evaluate_fold": 5, "aggregate": 0}):
    raise SystemExit("aggregate historical ledger prefix/append order differs")
identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                          value.st_mode, value.st_nlink, value.st_mtime_ns,
                          value.st_ctime_ns)
def bind_once(relative):
    target = root / relative; before = target.lstat()
    if (target.is_symlink() or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1
            or str(target) != str(target.resolve(strict=True))
            or not hasattr(os, "O_NOFOLLOW")):
        raise RuntimeError("file envelope differs: " + relative)
    fd = os.open(target, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd); chunks = []; digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk: break
            chunks.append(chunk); digest.update(chunk)
        closed = os.fstat(fd)
    finally:
        os.close(fd)
    after = target.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise RuntimeError("same-FD identity differs: " + relative)
    return b"".join(chunks), {
        "relative_path": relative, "absolute_path": str(target.resolve(strict=True)),
        "sha256": digest.hexdigest(), "size_bytes": before.st_size,
        "mode_octal": "0444", "nlink": 1,
        "physical_identity": {"device": before.st_dev, "inode": before.st_ino,
                              "size_bytes": before.st_size,
                              "mode_octal": format(
                                  stat.S_IMODE(before.st_mode), "04o"),
                              "nlink": before.st_nlink,
                              "mtime_ns": before.st_mtime_ns,
                              "ctime_ns": before.st_ctime_ns},
        "same_fd_o_nofollow_verified": True,
    }
def bind_wait(relative):
    last = "not attempted"
    for attempt in range(1, 21):
        try: return bind_once(relative)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            last = type(error).__name__ + ":" + str(error)
            print("V4G_HISTORY_NFS_RETRY stage=aggregate path=" + relative
                  + " attempt=" + str(attempt) + " diagnostic=" + last,
                  file=sys.stderr, flush=True)
            if attempt != 20: time.sleep(1.0)
    raise SystemExit("aggregate historical capture timeout: " + last)
if (not root.is_absolute() or root.is_symlink()
        or str(root) != str(root.resolve(strict=True))
        or path != root / "logs/aggregate.stdout"):
    raise SystemExit("aggregate historical path differs")
relative_paths = (["logs/aggregate.stdout", "logs/aggregate.stderr",
                   "aggregate/receipt.json", "barrier/barrier.json"]
                  + ["fold" + str(index) + "/fold.json" for index in range(5)])
captures = [bind_wait(relative) for relative in relative_paths]
raws = {relative: captured[0] for relative, captured in zip(relative_paths, captures)}
bindings = [captured[1] for captured in captures]
try:
    lines = raws[relative_paths[0]].decode("utf-8", errors="strict").splitlines()
except UnicodeDecodeError as error:
    raise SystemExit("aggregate structured stdout UTF-8 differs: " + str(error))
prefixes = ["V4G_CPU_GATE", "V4G_AGGREGATE_RUNTIME", "V4G_AGGREGATE_VERIFY",
            "V4G_CACHE_CLEANED"]
if len(lines) != 4 or [line.split("=", 1)[0] for line in lines] != prefixes:
    raise SystemExit("aggregate structured stdout exact4 differs")
try:
    values = [json.loads(line.split("=", 1)[1], object_pairs_hook=pairs,
                         parse_constant=nonfinite) for line in lines]
    receipt = json.loads(raws[relative_paths[2]], object_pairs_hook=pairs,
                         parse_constant=nonfinite)
except (json.JSONDecodeError, ValueError) as error:
    raise SystemExit("aggregate historical structured JSON differs: " + str(error))
runtime, verify = values[1], values[2]
unsigned = dict(receipt); receipt_digest = unsigned.pop("receipt_digest", None)
by_relative = {row["relative_path"]: row for row in bindings}
receipt_binding = by_relative[relative_paths[2]]
barrier_binding = by_relative[relative_paths[3]]
gate = verify.get("exposed_five_view_codec_development_gate")
if (values[0].get("role") != "aggregate"
        or runtime.get("receipt_sha256") != verify.get("aggregate_receipt_sha256")
        or runtime.get("receipt_digest") != verify.get("aggregate_receipt_digest")
        or runtime.get("exposed_five_view_codec_development_gate")
           is not verify.get("exposed_five_view_codec_development_gate")
        or verify.get("barrier_receipt_sha256") != expected_barrier
        or values[3].get("role") != "aggregate"
        or values[3].get("absent_after_cleanup") is not True
        or receipt.get("schema_version")
           != "semantic-anchor-vjepa2-role-directed-teacher-margin-exact5-receipt-v4g"
        or receipt.get("status")
           != "V4G_ROLE_DIRECTED_TEACHER_MARGIN_KNOWN_EXPOSED_DEVELOPMENT"
        or receipt_digest != object_sha(unsigned)
        or receipt_digest != verify.get("aggregate_receipt_digest")
        or receipt_binding["sha256"] != verify.get("aggregate_receipt_sha256")
        or receipt_binding["size_bytes"] != verify.get("aggregate_receipt_size_bytes")
        or barrier_binding["sha256"] != expected_barrier
        or type(gate) is not bool
        or receipt.get("metrics", {}).get(
            "exposed_five_view_codec_development_gate") is not gate
        or type(receipt.get("controller_barrier_receipt_binding")) is not dict
        or type(receipt.get("inner_receipts")) is not dict
        or type(receipt.get("fold_receipts")) is not dict
        or type(receipt.get("folds")) is not list
        or len(receipt.get("folds")) != 5):
    raise SystemExit("aggregate structured stdout semantic join differs")
record = {
    "stage": "aggregate", "files": bindings,
    "structured_values_sha256": object_sha(values),
    "receipt_digest": receipt_digest, "receipt_status": receipt.get("status"),
    "exposed_five_view_codec_development_gate": gate,
    "controller_barrier_receipt_binding_sha256": object_sha(
        receipt.get("controller_barrier_receipt_binding")),
    "inner_receipt_binding_ledger_sha256": object_sha(
        receipt.get("inner_receipts")),
    "fold_receipt_binding_ledger_sha256": object_sha(
        receipt.get("fold_receipts")),
    "fold_value_ledger_sha256": object_sha(receipt.get("folds")),
    "global_inner_barrier_sha256": object_sha(receipt.get("global_inner_barrier")),
}
ledger["aggregate"] = record
ledger["append_sequence"].append("aggregate")
ledger["captured_counts"]["aggregate"] = 1
new_encoded = base64.urlsafe_b64encode(canonical(ledger)).decode("ascii")
if len(new_encoded) > 98304:
    raise SystemExit("aggregate compact historical ledger exceeds argv safety bound")
print(verify["aggregate_receipt_sha256"] + ":" + verify["aggregate_receipt_digest"]
      + ":" + str(verify["aggregate_receipt_size_bytes"])
      + ":" + ("true" if verify[
          "exposed_five_view_codec_development_gate"] else "false")
      + ":" + new_encoded + ":" + object_sha(ledger))
PY
}

parse_phase_state_output() {
  [[ $# -eq 5 || $# -eq 6 ]] || \
    fail "parse_phase_state_output argument count differs"
  local fn_python_bin=$1 fn_line=$2 fn_kind fn_phase fn_path fn_history_sha
  if [[ $# -eq 5 ]]; then
    fn_kind=phase-state; fn_phase=$3; fn_path="$4/phase-state.json"; fn_history_sha=$5
  else
    fn_kind=$3; fn_phase=$4; fn_path=$5; fn_history_sha=$6
  fi
  "${fn_python_bin}" -I -S -B - "${fn_line}" "${fn_kind}" "${fn_phase}" \
    "${fn_path}" "${fn_history_sha}" <<'PY'
import json, re, sys
line, kind, phase, path, historical_sha = sys.argv[1:6]
def pairs(values):
    result = {}
    for key, value in values:
        if key in result: raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result
def nonfinite(value): raise ValueError("nonfinite JSON constant: " + value)
prefix = {"phase-state": "V4G_PHASE_STATE=", "success": "V4G_FINAL_SEAL=",
          "no-go": "V4G_GLOBAL_NO_GO_SEAL="}.get(kind)
if prefix is None or not line.startswith(prefix) or "\n" in line:
    raise SystemExit("controller seal structured output envelope differs")
try:
    value = json.loads(line.split("=", 1)[1], object_pairs_hook=pairs,
                       parse_constant=nonfinite)
except (json.JSONDecodeError, ValueError) as error:
    raise SystemExit("controller seal structured JSON differs: " + str(error))
if kind != "phase-state":
    if (value.get("seal") != path or value.get("branch") != kind
            or value.get("phase") != phase
            or value.get("append_only_historical_ledger_sha256") != historical_sha
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("seal_sha256"))) is None
            or re.fullmatch(r"[0-9a-f]{64}",
                            str(value.get("final_stable_tree_sha256"))) is None
            or type(value.get("final_exact_file_count")) is not int
            or value["final_exact_file_count"] <= 0):
        raise SystemExit("final-seal structured output semantic binding differs")
    print(":".join([value["seal_sha256"],
                    str(value["final_exact_file_count"]),
                    value["final_stable_tree_sha256"]]))
    raise SystemExit(0)
keys = ["file_sha256", "phase_state_digest",
        "controller_captured_artifacts_sha256",
        "pre_phase_state_files_sha256", "pre_phase_state_tree_sha256"]
if (value.get("path") != path or value.get("phase") != phase
        or value.get("append_only_historical_ledger_sha256") != historical_sha
        or type(value.get("size_bytes")) is not int
        or value["size_bytes"] <= 0
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value.get(key))) is None
               for key in keys)):
    raise SystemExit("phase-state structured output semantic binding differs")
print(":".join([value["file_sha256"], str(value["size_bytes"]),
                value["phase_state_digest"],
                value["controller_captured_artifacts_sha256"],
                value["pre_phase_state_files_sha256"],
                value["pre_phase_state_tree_sha256"],
                value["append_only_historical_ledger_sha256"]]))
PY
}

finish_branch() {
  [[ $# -eq 4 ]] || fail "finish_branch argument count differs"
  local fn_mode=$1 fn_phase=$2 fn_reason=$3 fn_allfold_oof0=$4
  local fn_parent_transport fn_parent_encoded fn_parent_sha
  local fn_child_output fn_phase_output fn_seal_output fn_seal_path fn_bad
  local fn_postseal_history fn_final_transport fn_final_seal_sha
  local fn_final_exact_count fn_final_stable_tree_sha
  fn_parent_transport="$(parent_stabilize_phase_inputs "${main_python_bin}" \
    "${main_run_root}" "${fn_phase}" "${main_historical_ledger_encoded}" \
    "${main_historical_ledger_sha}")"
  IFS=: read -r fn_parent_encoded fn_parent_sha <<<"${fn_parent_transport}"
  [[ -n "${fn_parent_encoded}" && "${fn_parent_sha}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "parent stabilized phase transport differs"
  recheck_holders "${fn_phase}-pre-seal-recheck"
  if ! fn_child_output="$(
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      -u GPU_DEVICE_ORDINAL \
      srun --jobid="${cpu_job}" --nodelist="${cpu_node}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${cpu_cpus}" --mem="${cpu_memory}" \
        --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${main_controller_source}" __run_seal_child \
          "${fn_mode}" "${main_release_root}" "${main_python_bin}" \
          "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
          "${main_v4d_receipt}" "${main_run_root}" "${cpu_job}" "${cpu_node}" \
          "${main_expected_controller_sha}" "${main_authority_pre}" \
          "${fn_phase}" "${fn_allfold_oof0}" "${fn_reason}" \
          "${fn_parent_encoded}" "${fn_parent_sha}" \
          "${main_historical_ledger_encoded}" "${main_historical_ledger_sha}"
  )"; then
    fail "${fn_phase} final seal child failed"
  fi
  [[ "$(printf '%s\n' "${fn_child_output}" | wc -l | tr -d ' ')" == 2 \
      && "${fn_child_output}" == *$'\n'* ]] || fail "seal child stdout exact2 differs"
  fn_phase_output=${fn_child_output%%$'\n'*}
  fn_seal_output=${fn_child_output#*$'\n'}
  [[ "${fn_seal_output}" != *$'\n'* ]] || fail "seal child final line differs"
  [[ "${fn_phase_output}" == V4G_PHASE_STATE=* ]] || fail "phase-state stdout differs"
  if [[ "${fn_mode}" == success ]]; then
    [[ "${fn_seal_output}" == V4G_FINAL_SEAL=* ]] || fail "success seal stdout differs"
    fn_seal_path="${main_run_root}/seal.json"
  else
    [[ "${fn_seal_output}" == V4G_GLOBAL_NO_GO_SEAL=* ]] || fail "no-go seal stdout differs"
    fn_seal_path="${main_run_root}/global-no-go-seal.json"
  fi
  fn_final_transport="$(parse_phase_state_output "${main_python_bin}" \
    "${fn_seal_output}" "${fn_mode}" "${fn_phase}" "${fn_seal_path}" \
    "${main_historical_ledger_sha}")"
  IFS=: read -r fn_final_seal_sha fn_final_exact_count \
    fn_final_stable_tree_sha <<<"${fn_final_transport}"
  [[ "${fn_final_seal_sha}" =~ ^[0-9a-f]{64}$ \
      && "${fn_final_exact_count}" =~ ^[1-9][0-9]*$ \
      && "${fn_final_stable_tree_sha}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "final seal structured transport differs"
  nfs_wait_path "${fn_seal_path}" file final-seal
  nfs_wait_path "${main_run_root}" directory final-run-root
  fn_bad="$(find "${main_run_root}" -type d ! -perm 0555 -print -quit)"
  [[ -z "${fn_bad}" ]] || fail "final directory mode closure differs"
  fn_bad="$(find "${main_run_root}" -type f \( ! -perm 0444 -o ! -links 1 \) -print -quit)"
  [[ -z "${fn_bad}" ]] || fail "final file mode/link closure differs"
  fn_bad="$(find "${main_run_root}" -mindepth 1 ! -type f ! -type d -print -quit)"
  [[ -z "${fn_bad}" ]] || fail "final tree contains symlink/special member"
  fn_postseal_history="$(parent_stabilize_phase_inputs "${main_python_bin}" \
    "${main_run_root}" "${fn_phase}" "${main_historical_ledger_encoded}" \
    "${main_historical_ledger_sha}" post-seal "$(basename -- "${fn_seal_path}")" \
    "${fn_final_seal_sha}" "${fn_final_exact_count}" \
    "${fn_final_stable_tree_sha}")"
  [[ "${fn_postseal_history}" == V4G_PARENT_HISTORICAL_POSTSEAL=* \
      && "${fn_postseal_history}" != *$'\n'* \
      && "${fn_postseal_history}" == *\"historical_ledger_sha256\":\"${main_historical_ledger_sha}\"* \
      && "${fn_postseal_history}" == *\"same_parent_root_and_file_physical_identity_reverified\":true* \
      && "${fn_postseal_history}" == *\"exact_final_tree_and_child_seal_sha_reverified\":true* \
      && "${fn_postseal_history}" == *\"final_stable_tree_sha256\":\"${fn_final_stable_tree_sha}\"* ]] || \
    fail "parent historical post-seal bracket differs"
  printf '%s\n' "${fn_phase_output}" "${fn_seal_output}"
  if [[ "${fn_mode}" == success ]]; then
    printf 'V4G_EXACT5_PARALLEL_COMPLETE run_root=%s\n' "${main_run_root}"
  else
    printf 'V4G_GLOBAL_NO_GO_COMPLETE phase=%s allfold_oof_exact0=%s run_root=%s\n' \
      "${fn_phase}" "${fn_allfold_oof0}" "${main_run_root}"
  fi
  exit 0
}

if [[ "${1:-}" == __run_preflight_child ]]; then
  shift; run_preflight_child "$@"; exit 0
elif [[ "${1:-}" == __run_train_child ]]; then
  shift; run_train_child "$@"; exit 0
elif [[ "${1:-}" == __run_barrier_child ]]; then
  shift; run_barrier_child "$@"; exit 0
elif [[ "${1:-}" == __run_evaluate_child ]]; then
  shift; run_evaluate_child "$@"; exit 0
elif [[ "${1:-}" == __run_aggregate_child ]]; then
  shift; run_aggregate_child "$@"; exit 0
elif [[ "${1:-}" == __run_seal_child ]]; then
  shift; run_seal_child "$@"; exit 0
fi

[[ $# -eq 8 ]] || fail \
  "usage: $0 RELEASE PYTHON FEATURE_ROOT V4A_RECEIPT V4C_FRONTIER_RECEIPT V4D_RECEIPT FRESH_RUN_ROOT EXPECTED_CONTROLLER_SHA256"
readonly main_release_root=$1
readonly main_python_bin=$2
readonly main_feature_root=$3
readonly main_v4a_receipt=$4
readonly main_v4c_receipt=$5
readonly main_v4d_receipt=$6
readonly main_run_root=$7
readonly main_expected_controller_sha=$8
main_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
readonly main_controller_source

[[ "${main_expected_controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller SHA argument differs"
require_plain_file "${main_controller_source}" "${main_expected_controller_sha}" 555 detached-controller
[[ "${main_controller_source}" != "${main_release_root%/}/"* ]] || \
  fail "controller must remain detached from release"
require_plain_file "${main_python_bin}" "${expected_python_sha256}" 755 pinned-python
require_release "${main_release_root}" "${main_python_bin}"
require_plain_file "${main_feature_root}/feature_extraction_receipt.json" \
  "${expected_feature_receipt_sha256}" 444 feature-receipt
require_plain_file "${main_v4a_receipt}" "${expected_v4a_receipt_sha256}" 444 v4a-receipt
require_plain_file "${main_v4c_receipt}" "${expected_v4c_frontier_receipt_sha256}" 444 v4c-receipt
require_plain_file "${main_v4d_receipt}" "${expected_v4d_receipt_sha256}" 444 v4d-receipt
[[ "${main_run_root}" == /* && "${main_run_root}" != / \
    && ! -e "${main_run_root}" && ! -L "${main_run_root}" \
    && "${main_run_root}" == "$(readlink -m -- "${main_run_root}")" ]] || \
  fail "run root is not fresh/absolute/canonical/safe"
main_run_parent="$(dirname -- "${main_run_root}")"; readonly main_run_parent
[[ -d "${main_run_parent}" && ! -L "${main_run_parent}" && -w "${main_run_parent}" \
    && "${main_run_parent}" == "$(readlink -f -- "${main_run_parent}")" ]] || \
  fail "run parent differs"
main_authority_pre="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
  "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
  "${main_v4d_receipt}" "${main_controller_source}" \
  "${main_expected_controller_sha}")"
readonly main_authority_pre
[[ "${main_authority_pre}" =~ ^[0-9a-f]{64}:[0-9a-f]{64}:[0-9a-f]{64}:[0-9a-f]{64}$ ]] || \
  fail "initial authority snapshot differs"

recheck_holders preflight
if ! main_preflight_output="$(
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    -u GPU_DEVICE_ORDINAL \
    srun --jobid="${preflight_job}" --nodelist="${preflight_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${preflight_cpus}" --mem="${preflight_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_preflight_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_receipt}" "${main_v4d_receipt}" \
        "${preflight_job}" "${preflight_node}" "${main_expected_controller_sha}" \
        "${main_authority_pre}"
)"; then
  fail "CPU preflight failed"
fi
[[ "${main_preflight_output}" == V4G_PREFLIGHT_PASS=* \
    && "$(printf '%s\n' "${main_preflight_output}" | wc -l | tr -d ' ')" == 1 ]] || \
  fail "CPU preflight stdout differs"
[[ ! -e "${main_run_root}" && ! -L "${main_run_root}" ]] || \
  fail "preflight created persistent run root"
[[ "$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
  "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
  "${main_v4d_receipt}" "${main_controller_source}" \
  "${main_expected_controller_sha}")" == "${main_authority_pre}" ]] || \
  fail "preflight changed authority"
recheck_holders post-preflight

mkdir "${main_run_root}"; mkdir "${main_run_root}/logs"
chmod 0700 "${main_run_root}" "${main_run_root}/logs"
main_historical_init="$(
  "${main_python_bin}" -I -S -B - "${main_run_root}/launch-plan.json" \
    "${main_expected_controller_sha}" "${main_authority_pre}" \
    "${main_preflight_output}" <<'PY'
from pathlib import Path
import base64, hashlib, json, os, stat, sys
path = Path(sys.argv[1])
preflight = json.loads(sys.argv[4].split("=", 1)[1])
object_sha = lambda value: hashlib.sha256(json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).hexdigest()
payload = {
    "schema_version": "v4g-exact5-role-directed-launch-plan-v1",
    "controller_sha256": sys.argv[2], "authority_snapshot": sys.argv[3],
    "cpu_preflight": preflight,
    "source_authority": {
        "release_tree_sha256": "e4e158e064ceb181345673c86a8fb275436ddd25edc74a3e7ef8f1c31d4f16ff",
        "release_manifest_sha256": "d1e0c42904057e14d47c87e746c32db375ba6ee6b006f813ea91cdb2daae4882",
        "release_manifest_digest": "91adc268b8b86f083c36aa043e5c6f10956e720b03a3b5ebf1022b3775ac46a4",
        "runtime_sha256": "38b2cbecaf022e203ccf09e6808661013f4f23dee0d02ffa1756e24d0c167cf9",
        "runtime_test_sha256": "7fe6b42208f77171f99d44d5a9fc9eae58c3bb2d4663ca016e9b154a4d3c4996",
        "python_sha256": "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",
    },
    "fixed_no_fallback": True,
    "workers": [
        {"fold": 0, "job": 143808, "node": "auh7-1b-gpu-315"},
        {"fold": 1, "job": 143812, "node": "auh7-1b-gpu-293"},
        {"fold": 2, "job": 143811, "node": "auh7-1b-gpu-306"},
        {"fold": 3, "job": 143808, "node": "auh7-1b-gpu-233"},
        {"fold": 4, "job": 143808, "node": "auh7-1b-gpu-268"},
    ],
    "worker_request": {"nodes": 1, "ntasks": 1, "cpus": 8,
                       "memory": "12G", "gres": "gpu:mi210:1",
                       "overlap": True, "exact": True},
    "state_machine": ["train-fold-exact5", "verify-inner-barrier-exact1",
                      "evaluate-fold-exact5", "aggregate-exact1",
                      "branch-seal-exact1"],
    "train_fold_outputs": ["preselection.pt", "fixed1200.pt", "inner.json"],
    "train_fold_oof_semantic_tensor_read_count": 0,
    "barrier_before_any_evaluate": True,
    "any_train_inner_or_barrier_failure_forbids_all_evaluate_and_aggregate": True,
    "allfold_oof_exact0_on_train_inner_or_barrier_no_go": True,
    "controller_captures_inner_barrier_and_fold_receipt_shas": True,
    "official_controller_cli_caller_supplied_inner_barrier_or_fold_sha": False,
    "nfs_postseal_bounded_retry": {"attempts": 20, "seconds_per_attempt": 1},
    "same_fd_o_nofollow_final_tree_seal": True,
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                 ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
             0o400)
with os.fdopen(fd, "w+b") as handle:
    os.fchmod(handle.fileno(), 0o444); handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    written = os.fstat(handle.fileno()); handle.seek(0); readback = handle.read()
    closed = os.fstat(handle.fileno())
after = path.lstat()
identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode,
                          value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)
if (readback != raw or not (identity(written) == identity(closed) == identity(after))
        or path.is_symlink() or (after.st_mode & 0o7777) != 0o444 or after.st_nlink != 1):
    raise SystemExit("launch-plan create-only same-FD seal differs")
binding = {
    "relative_path": "launch-plan.json",
    "absolute_path": str(path.resolve(strict=True)),
    "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": after.st_size,
    "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
    "nlink": after.st_nlink,
    "physical_identity": {"device": after.st_dev, "inode": after.st_ino,
                          "size_bytes": after.st_size,
                          "mode_octal": format(stat.S_IMODE(after.st_mode), "04o"),
                          "nlink": after.st_nlink, "mtime_ns": after.st_mtime_ns,
                          "ctime_ns": after.st_ctime_ns},
    "same_fd_o_nofollow_verified": True,
}
root_info = path.parent.lstat()
ledger = {
    "schema_version": "v4g-controller-append-only-historical-ledger-v1",
    "run_root": str(path.parent.resolve(strict=True)),
    "run_root_physical_identity": {"device": root_info.st_dev,
                                   "inode": root_info.st_ino},
    "launch_plan": {"file_binding": binding, "value": payload},
    "train_folds": [], "barrier": None, "evaluate_folds": [],
    "aggregate": None, "append_sequence": ["launch-plan"],
    "captured_counts": {"launch_plan": 1, "train": 0, "barrier": 0,
                        "evaluate_fold": 0, "aggregate": 0},
}
encoded = base64.urlsafe_b64encode(json.dumps(
    ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    allow_nan=False,
).encode("ascii")).decode("ascii")
if len(encoded) > 98304:
    raise SystemExit("initial compact historical ledger exceeds argv safety bound")
print(encoded + ":" + object_sha(ledger))
PY
 )"
IFS=: read -r main_historical_ledger_encoded main_historical_ledger_sha \
  <<<"${main_historical_init}"
[[ -n "${main_historical_ledger_encoded}" \
    && "${main_historical_ledger_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "launch-plan historical ledger initialization differs"
unset main_historical_init

recheck_holders pre-train
main_train_pids=(); main_train_rcs=()
for main_index in 0 1 2 3 4; do
  main_stdout="${main_run_root}/logs/train-fold${main_index}.stdout"
  main_stderr="${main_run_root}/logs/train-fold${main_index}.stderr"
  (
    set -o noclobber
    exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      -u GPU_DEVICE_ORDINAL \
      srun --jobid="${worker_jobs[${main_index}]}" \
        --nodelist="${worker_nodes[${main_index}]}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${worker_cpus}" --mem="${worker_memory}" \
        --gres=gpu:mi210:1 --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${main_controller_source}" __run_train_child \
          "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
          "${main_v4a_receipt}" "${main_v4c_receipt}" "${main_v4d_receipt}" \
          "${main_run_root}/fold${main_index}" "${main_index}" \
          "${worker_jobs[${main_index}]}" "${worker_nodes[${main_index}]}" \
          "${main_expected_controller_sha}" "${main_authority_pre}" \
        >"${main_stdout}" 2>"${main_stderr}"
  ) &
  main_train_pids[${main_index}]=$!
done
main_train_failure=0
for main_index in 0 1 2 3 4; do
  main_train_rcs[${main_index}]=0
  wait "${main_train_pids[${main_index}]}" || main_train_rcs[${main_index}]=$?
  (( main_train_rcs[${main_index}] == 0 )) || main_train_failure=1
  seal_log_pair "${main_run_root}/logs/train-fold${main_index}.stdout" \
    "${main_run_root}/logs/train-fold${main_index}.stderr" "train-fold${main_index}"
done
if (( main_train_failure != 0 )); then
  finish_branch no-go TRAIN_FAILURE_ALL_OOF0 "train_rcs=${main_train_rcs[*]}" true
fi
main_inner_shas=(); main_inner_digests=(); main_inner_sizes=(); main_inner_passes=()
main_preselection_shas=(); main_preselection_sizes=()
main_fixed1200_shas=(); main_fixed1200_sizes=(); main_train_bindings=()
for main_index in 0 1 2 3 4; do
  if ! main_capture="$(parse_train_log "${main_python_bin}" \
      "${main_run_root}/logs/train-fold${main_index}.stdout" "${main_index}" \
      "${main_run_root}" "${main_historical_ledger_encoded}" \
      "${main_historical_ledger_sha}")"; then
    finish_branch no-go TRAIN_VERIFICATION_FAILURE_ALL_OOF0 \
      "train_log_parse_fold=${main_index}" true
  fi
  IFS=: read -r main_inner_shas[${main_index}] main_inner_digests[${main_index}] \
    main_inner_sizes[${main_index}] main_inner_passes[${main_index}] \
    main_preselection_shas[${main_index}] main_preselection_sizes[${main_index}] \
    main_fixed1200_shas[${main_index}] main_fixed1200_sizes[${main_index}] \
    main_next_historical_encoded main_next_historical_sha \
    <<<"${main_capture}"
  [[ -n "${main_next_historical_encoded}" \
      && "${main_next_historical_sha}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "train historical ledger append output differs"
  main_historical_ledger_encoded=${main_next_historical_encoded}
  main_historical_ledger_sha=${main_next_historical_sha}
  main_train_bindings[${main_index}]="${main_inner_shas[${main_index}]}:${main_inner_sizes[${main_index}]}:${main_preselection_shas[${main_index}]}:${main_preselection_sizes[${main_index}]}:${main_fixed1200_shas[${main_index}]}:${main_fixed1200_sizes[${main_index}]}"
done
[[ "$(printf '%s\n' "${main_inner_shas[@]}" | sort -u | wc -l | tr -d ' ')" == 5 ]] || \
  fail "captured inner receipt SHAs are not exact-five distinct"
for main_index in 0 1 2 3 4; do
  if [[ "${main_inner_passes[${main_index}]}" != true ]]; then
    finish_branch no-go INNER_NO_GO_ALL_OOF0 "inner_fail_fold=${main_index}" true
  fi
done
[[ "$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
  "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
  "${main_v4d_receipt}" "${main_controller_source}" \
  "${main_expected_controller_sha}")" == "${main_authority_pre}" ]] || \
  fail "train stage changed authority"

recheck_holders pre-barrier
mkdir "${main_run_root}/barrier"; chmod 0700 "${main_run_root}/barrier"
main_barrier_stdout="${main_run_root}/logs/barrier.stdout"
main_barrier_stderr="${main_run_root}/logs/barrier.stderr"
main_barrier_rc=0
(
  set -o noclobber
  exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    -u GPU_DEVICE_ORDINAL \
    srun --jobid="${worker_jobs[${barrier_worker_index}]}" \
      --nodelist="${worker_nodes[${barrier_worker_index}]}" --nodes=1 --ntasks=1 \
      --cpus-per-task="${worker_cpus}" --mem="${worker_memory}" \
      --gres=gpu:mi210:1 --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_barrier_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_receipt}" "${main_v4d_receipt}" \
        "${main_run_root}" "${worker_jobs[${barrier_worker_index}]}" \
        "${worker_nodes[${barrier_worker_index}]}" "${main_expected_controller_sha}" \
        "${main_authority_pre}" "${main_train_bindings[@]}" \
      >"${main_barrier_stdout}" 2>"${main_barrier_stderr}"
) || main_barrier_rc=$?
seal_log_pair "${main_barrier_stdout}" "${main_barrier_stderr}" barrier
if (( main_barrier_rc != 0 )); then
  finish_branch no-go BARRIER_FAILURE_ALL_OOF0 "barrier_rc=${main_barrier_rc}" true
fi
if ! main_barrier_capture="$(parse_barrier_log "${main_python_bin}" \
    "${main_barrier_stdout}" "${main_run_root}" \
    "${main_historical_ledger_encoded}" "${main_historical_ledger_sha}")"; then
  finish_branch no-go BARRIER_FAILURE_ALL_OOF0 barrier_log_parse true
fi
IFS=: read -r main_barrier_sha main_barrier_digest main_barrier_size \
  main_next_historical_encoded main_next_historical_sha \
  <<<"${main_barrier_capture}"
[[ -n "${main_next_historical_encoded}" \
    && "${main_next_historical_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "barrier historical ledger append output differs"
main_historical_ledger_encoded=${main_next_historical_encoded}
main_historical_ledger_sha=${main_next_historical_sha}
readonly main_barrier_sha main_barrier_digest main_barrier_size
[[ "$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
  "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
  "${main_v4d_receipt}" "${main_controller_source}" \
  "${main_expected_controller_sha}")" == "${main_authority_pre}" ]] || \
  fail "barrier changed authority"

recheck_holders pre-evaluate
main_eval_pids=(); main_eval_rcs=()
for main_index in 0 1 2 3 4; do
  main_stdout="${main_run_root}/logs/evaluate-fold${main_index}.stdout"
  main_stderr="${main_run_root}/logs/evaluate-fold${main_index}.stderr"
  (
    set -o noclobber
    exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
      -u GPU_DEVICE_ORDINAL \
      srun --jobid="${worker_jobs[${main_index}]}" \
        --nodelist="${worker_nodes[${main_index}]}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${worker_cpus}" --mem="${worker_memory}" \
        --gres=gpu:mi210:1 --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${main_controller_source}" __run_evaluate_child \
          "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
          "${main_v4a_receipt}" "${main_v4c_receipt}" "${main_v4d_receipt}" \
          "${main_run_root}" "${main_index}" "${main_run_root}/barrier/barrier.json" \
          "${main_barrier_sha}" "${main_barrier_digest}" \
          "${main_inner_shas[${main_index}]}" "${main_barrier_size}" \
          "${worker_jobs[${main_index}]}" \
          "${worker_nodes[${main_index}]}" "${main_expected_controller_sha}" \
          "${main_authority_pre}" "${main_train_bindings[@]}" \
        >"${main_stdout}" 2>"${main_stderr}"
  ) &
  main_eval_pids[${main_index}]=$!
done
main_eval_failure=0
for main_index in 0 1 2 3 4; do
  main_eval_rcs[${main_index}]=0
  wait "${main_eval_pids[${main_index}]}" || main_eval_rcs[${main_index}]=$?
  (( main_eval_rcs[${main_index}] == 0 )) || main_eval_failure=1
  seal_log_pair "${main_run_root}/logs/evaluate-fold${main_index}.stdout" \
    "${main_run_root}/logs/evaluate-fold${main_index}.stderr" "evaluate-fold${main_index}"
done
if (( main_eval_failure != 0 )); then
  finish_branch no-go EVALUATE_FAILURE "evaluate_rcs=${main_eval_rcs[*]}" false
fi
main_fold_shas=(); main_fold_digests=(); main_fold_sizes=(); main_fold_bindings=()
for main_index in 0 1 2 3 4; do
  if ! main_capture="$(parse_evaluate_log "${main_python_bin}" \
      "${main_run_root}/logs/evaluate-fold${main_index}.stdout" "${main_index}" \
      "${main_barrier_sha}" "${main_inner_shas[${main_index}]}" \
      "${main_run_root}" "${main_historical_ledger_encoded}" \
      "${main_historical_ledger_sha}")"; then
    finish_branch no-go EVALUATE_FAILURE "evaluate_log_parse_fold=${main_index}" false
  fi
  IFS=: read -r main_fold_shas[${main_index}] main_fold_digests[${main_index}] \
    main_fold_sizes[${main_index}] main_next_historical_encoded \
    main_next_historical_sha <<<"${main_capture}"
  [[ -n "${main_next_historical_encoded}" \
      && "${main_next_historical_sha}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "evaluate historical ledger append output differs"
  main_historical_ledger_encoded=${main_next_historical_encoded}
  main_historical_ledger_sha=${main_next_historical_sha}
  main_fold_bindings[${main_index}]="${main_fold_shas[${main_index}]}:${main_fold_sizes[${main_index}]}"
done
[[ "$(printf '%s\n' "${main_fold_shas[@]}" | sort -u | wc -l | tr -d ' ')" == 5 ]] || \
  fail "captured fold receipt SHAs are not exact-five distinct"
[[ "$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
  "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
  "${main_v4d_receipt}" "${main_controller_source}" \
  "${main_expected_controller_sha}")" == "${main_authority_pre}" ]] || \
  fail "evaluate changed authority"

recheck_holders pre-aggregate
mkdir "${main_run_root}/aggregate"; chmod 0700 "${main_run_root}/aggregate"
main_aggregate_stdout="${main_run_root}/logs/aggregate.stdout"
main_aggregate_stderr="${main_run_root}/logs/aggregate.stderr"
main_aggregate_rc=0
(
  set -o noclobber
  exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
    -u GPU_DEVICE_ORDINAL \
    srun --jobid="${cpu_job}" --nodelist="${cpu_node}" --nodes=1 --ntasks=1 \
      --cpus-per-task="${cpu_cpus}" --mem="${cpu_memory}" --gres=none \
      --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_aggregate_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_receipt}" "${main_v4d_receipt}" \
        "${main_run_root}" "${main_run_root}/barrier/barrier.json" \
        "${main_barrier_sha}" "${main_barrier_digest}" "${main_barrier_size}" \
        "${cpu_job}" "${cpu_node}" \
        "${main_expected_controller_sha}" "${main_authority_pre}" \
        "${main_fold_bindings[@]}" "${main_train_bindings[@]}" \
      >"${main_aggregate_stdout}" 2>"${main_aggregate_stderr}"
) || main_aggregate_rc=$?
seal_log_pair "${main_aggregate_stdout}" "${main_aggregate_stderr}" aggregate
if (( main_aggregate_rc != 0 )); then
  finish_branch no-go AGGREGATE_FAILURE "aggregate_rc=${main_aggregate_rc}" false
fi
if ! main_aggregate_capture="$(parse_aggregate_log "${main_python_bin}" \
    "${main_aggregate_stdout}" "${main_barrier_sha}" "${main_run_root}" \
    "${main_historical_ledger_encoded}" "${main_historical_ledger_sha}")"; then
  finish_branch no-go AGGREGATE_FAILURE aggregate_log_parse false
fi
IFS=: read -r main_aggregate_sha main_aggregate_digest main_aggregate_size \
  main_aggregate_gate main_next_historical_encoded main_next_historical_sha \
  <<<"${main_aggregate_capture}"
[[ -n "${main_next_historical_encoded}" \
    && "${main_next_historical_sha}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "aggregate historical ledger append output differs"
main_historical_ledger_encoded=${main_next_historical_encoded}
main_historical_ledger_sha=${main_next_historical_sha}
readonly main_aggregate_sha main_aggregate_digest main_aggregate_size main_aggregate_gate
[[ "$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
  "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_receipt}" \
  "${main_v4d_receipt}" "${main_controller_source}" \
  "${main_expected_controller_sha}")" == "${main_authority_pre}" ]] || \
  fail "aggregate changed authority"
finish_branch success SUCCESS exact5_barrier_evaluate_aggregate_verified false

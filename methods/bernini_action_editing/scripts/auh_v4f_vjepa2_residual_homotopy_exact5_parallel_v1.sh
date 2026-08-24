#!/usr/bin/env bash
# INTENTIONAL NO-GO controller candidate for five independent ordinary exact1
# MI210 v4-F fold workers followed by a branch-aware CPU-only postflight.  This
# controller never
# submits an allocation and has no automatic holder/node fallback.
#
# The normal entry point requires the exact sealed runtime/test/release/Python
# pins and receipt postflight below before any persistent run-root mutation.

set -Eeuo pipefail
umask 077

readonly tag=v4f-vjepa2-residual-homotopy-exact5-parallel-v1
readonly release_sealed=true
readonly python_pin_sealed=true
readonly controller_contract_complete=true
readonly placeholder_sha256=0000000000000000000000000000000000000000000000000000000000000000

# Sealed pins.  Do not replace these independently: runtime, tests, exact
# release tree, CLI, fold schema, aggregate schema, and postflight are frozen
# as one executable contract.
readonly expected_release_manifest_sha256=7e9a0fbcbce4743a32f53335a3f268eac7c7f579614feb91b6eaa35f44b2a471
readonly expected_release_manifest_digest=feae69e528e28a43ddca93ae8eb06883a94d794ec1a33ce0c5ab6bc61279f19b
readonly expected_release_tree_sha256=5282306dcb95c76e48549395d0fb030775be3990b23a043cc34a3e6d3123036d
readonly expected_runtime_sha256=97cd77e64a4dfaf3036e6c50a5b85060fd616f87371e5d967e69db1170466d74
readonly expected_runtime_test_sha256=15b70d8f56340f553f8f6d907cc5c20e25d2b0c0ed88d0a81f2b66f5bd3ac319
readonly expected_v4e_burned_runtime_sha256=4d8b518122a01a294d6190732da14da9614b1f041cf72c1ca69e4574b72ee96a
readonly expected_fold_schema=semantic-anchor-vjepa2-residual-homotopy-fold-receipt-v4f
readonly expected_aggregate_schema=semantic-anchor-vjepa2-residual-homotopy-exact5-receipt-v4f
readonly expected_aggregate_status=V4F_RESIDUAL_HOMOTOPY_KNOWN_EXPOSED_DEVELOPMENT
readonly expected_inner_no_go_status=V4F_INNER_NO_GO_OOF_UNREAD
readonly expected_checkpoint_schema=semantic-anchor-vjepa2-residual-homotopy-fold-checkpoint-v4f

readonly expected_python_sha256=8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a
readonly expected_feature_authority_sha256=74fe161f08eb92746ce22f00b50cca65f85ef05e4a4a6a5d0c7fb85867e94233
readonly expected_v2_runtime_sha256=46927772a1861354ad5edeb2072ae9b1b505d235de7c2615fb11a6648f2bddca
readonly expected_v4a_runtime_sha256=e7e755a430b79c34fdc86f5fceaba8a9f69c66dd1e66b47c8f4115eac5265973
readonly expected_extractor_sha256=720033ac069dd1ee33463d2c439199cfdce3a1c595d4252b7f395e68c56e1cfc
readonly expected_v4c_runtime_sha256=d286c23b0626aae2161deb12a465e8614fa1462dc74f3ab9b8afd88befee1cef
readonly expected_v4d_runtime_sha256=20934925e6c9bff364e6d00996f3713c9a2b254cf3ecfe0b506f03df35e146dc

readonly expected_feature_receipt_sha256=895fd7e9267c82477ffc11fbc1a11fdd89b276687d87c8e82e7d85d7cf62b54a
readonly expected_v4a_receipt_sha256=568ef85d9812bcc2a771952e1806392c80f8248f5597dd32e4c95e7e1f5a3fa2
readonly expected_v4a_receipt_digest=f33d72320905aba135a2bb8729782cf5c89e6eee81fe1bd88aa8d24e1b585a86
readonly expected_v4c_frontier_receipt_sha256=8b7a38d0fd9e8b789cb47b1be58a0e35615f5f4dae54df956de4103f00e5fef9
readonly expected_v4c_frontier_receipt_digest=376a98dc74e30ab80a277c8866028677d56ba894073d195612a0edb0bbd74f17
readonly expected_v4d_receipt_sha256=53910bcb71ce02a193bd47e44c3a97de0ee24f431576db64a763637447720b6f
readonly expected_v4d_receipt_digest=45d2ae7c45f1db8ccee9b14ba8a7543cfd1ff0d311128472ae116d6befa92f9c

# Frozen no-fallback map after accounting for live four-GPU steps on nodes
# 233/268/292.  Node 292 is intentionally unused by v4-F.
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
readonly aggregate_job=143811
readonly aggregate_node=auh7-1b-gpu-306
readonly aggregate_cpus=8
readonly aggregate_memory=12G

readonly v4e_burned_runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4e_alt.py
readonly runtime_relative=methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py
readonly runtime_test_relative=methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py
readonly release_manifest_relative=release-manifest-v4f.json
readonly expected_release_manifest_schema=v4f-vjepa2-residual-homotopy-detached-release-manifest-v1
readonly expected_release_manifest_status=V4F_DETACHED_RELEASE_MANIFEST_SEALED
readonly -a release_relative_files=(
  methods/bernini_action_editing/semantic_action_cvae_canary_v1.py
  methods/bernini_action_editing/semantic_anchor_action_sequence_vae_v2.py
  methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py
  methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py
  methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py
  methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py
  "${v4e_burned_runtime_relative}"
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
    '{"controller":"v4f-vjepa2-residual-homotopy-exact5-parallel-v1","intentional_no_go":false,"launch_performed":false,"release_sealed":true,"python_pin_sealed":true,"controller_contract_complete":true,"runtime_pin_final":true,"runtime_test_pin_final":true,"release_manifest_sha256":"7e9a0fbcbce4743a32f53335a3f268eac7c7f579614feb91b6eaa35f44b2a471","release_manifest_digest":"feae69e528e28a43ddca93ae8eb06883a94d794ec1a33ce0c5ab6bc61279f19b","release_manifest_status":"V4F_DETACHED_RELEASE_MANIFEST_SEALED","release_tree_sha256":"5282306dcb95c76e48549395d0fb030775be3990b23a043cc34a3e6d3123036d","release_exact_files":10,"fixed_no_fallback":true,"preflight_request":{"job":143811,"node":"auh7-1b-gpu-306","nodes":1,"ntasks":1,"cpus":4,"memory":"4G","gres":"none","overlap":true,"exact":true,"normal_tests":30,"optimized_tests":30,"skips":0,"before_first_persistent_mkdir":true},"worker_request":{"nodes":1,"ntasks":1,"cpus":8,"memory":"12G","gres":"gpu:mi210:1","overlap":true,"exact":true},"aggregate_request":{"job":143811,"node":"auh7-1b-gpu-306","nodes":1,"ntasks":1,"cpus":8,"memory":"12G","gres":"none","overlap":true,"exact":true},"fold_mapping":[{"fold":0,"job":143808,"node":"auh7-1b-gpu-315"},{"fold":1,"job":143812,"node":"auh7-1b-gpu-293"},{"fold":2,"job":143811,"node":"auh7-1b-gpu-306"},{"fold":3,"job":143808,"node":"auh7-1b-gpu-233"},{"fold":4,"job":143808,"node":"auh7-1b-gpu-268"}],"runtime_cli":{"train":"train-fold --fold-index N --fold-root ROOT","aggregate":"aggregate --fold-root ROOT repeated exactly five times --output RECEIPT"},"pass_branch":{"fold_exact_files":["preselection.pt","selected.pt","fold.json"],"success_preseal_exact":29,"success_postseal_exact":30},"inner_no_go_branch":{"fold_exact_files":["preselection.pt","fold.json"],"aggregate_forbidden":true,"partial_pass_fold_oof_may_have_been_read":true,"global_exact5_not_evaluated":true,"preseal_exact_formula":"21+p","postseal_exact_formula":"22+p"},"launch_or_remote_action_performed":false}'
  exit 0
fi

# These sealed execution gates precede argument validation and every command
# that could inspect an authority or mutate state.
[[ "${release_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: v4-F release/runtime/tests/manifest contract is not sealed"
[[ "${python_pin_sealed}" == true ]] || \
  fail "INTENTIONAL NO-GO: pinned AUH Python entity is not sealed"
[[ "${controller_contract_complete}" == true ]] || \
  fail "INTENTIONAL NO-GO: v4-F controller dual-branch postflight contract is incomplete"
[[ "${expected_release_tree_sha256}" != "${placeholder_sha256}" \
    && "${expected_release_manifest_sha256}" != "${placeholder_sha256}" \
    && "${expected_runtime_sha256}" != "${placeholder_sha256}" \
    && "${expected_runtime_test_sha256}" != "${placeholder_sha256}" ]] || \
  fail "INTENTIONAL NO-GO: v4-F source/tree/manifest pins are placeholders"

# Everything below is the frozen executable contract.

require_plain_file() {
  local fn_path=$1 fn_expected_sha=$2 fn_expected_mode=$3 fn_label=$4 fn_expected_size=${5:-}
  [[ "${fn_path}" == /* && -f "${fn_path}" && ! -L "${fn_path}" ]] || \
    fail "${fn_label} is not an absolute plain file"
  [[ "$(stat -c '%a:%h' "${fn_path}")" == "${fn_expected_mode}:1" ]] || \
    fail "${fn_label} mode/link differs"
  [[ "$(sha256_file "${fn_path}")" == "${fn_expected_sha}" ]] || \
    fail "${fn_label} SHA differs"
  if [[ -n "${fn_expected_size}" ]]; then
    [[ "$(stat -c %s "${fn_path}")" == "${fn_expected_size}" ]] || \
      fail "${fn_label} size differs"
  fi
}

require_release() {
  [[ $# -eq 2 ]] || fail "require release argument count differs"
  local fn_root=$1 fn_python_bin=$2 fn_validation
  if ! fn_validation="$("${fn_python_bin}" -I -S -B - \
      "${fn_root}" "${expected_release_tree_sha256}" \
      "${expected_release_manifest_digest}" "${expected_release_manifest_schema}" \
      "${expected_release_manifest_status}" \
      "${release_relative_files[@]}" "${release_expected_shas[@]}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

root = Path(sys.argv[1]); expected_tree = sys.argv[2]
expected_manifest_digest, manifest_schema, manifest_status = sys.argv[3:6]
relative_files = sys.argv[6:16]; expected_shas = sys.argv[16:26]
if (
    len(relative_files) != 10 or len(expected_shas) != 10
    or len(set(relative_files)) != 10
    or not root.is_absolute() or root.is_symlink()
    or str(root) != str(root.resolve(strict=True))
):
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
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_nlink != 1 or before.st_size <= 0
    ):
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
    if (
        path.is_symlink() or actual != expected_sha
        or not (identity(before) == identity(opened) == identity(closed) == identity(after))
    ):
        raise SystemExit(label + " same-FD SHA/identity differs")
    return b"".join(chunks), before, actual

all_members = list(root.rglob("*"))
if any(
    path.is_symlink()
    or (not path.is_dir() and not path.is_file())
    for path in all_members
):
    raise SystemExit("release contains symlink or special member")
actual_directories = {
    path.relative_to(root).as_posix() for path in all_members if path.is_dir()
}
actual_files = {
    path.relative_to(root).as_posix() for path in all_members if path.is_file()
}
if actual_directories != expected_directories or actual_files != set(relative_files):
    raise SystemExit("release exact10 tree membership differs")
if any(stat.S_IMODE((root / relative).lstat().st_mode) != 0o555
       for relative in expected_directories):
    raise SystemExit("release directory mode differs")
rows = []; raw_by_relative = {}
for relative, expected_sha in zip(relative_files, expected_shas):
    raw, info, actual = read_same_fd(
        root / relative, expected_sha, "release " + relative
    )
    raw_by_relative[relative] = raw
    rows.append({
        "path": relative, "sha256": actual, "size_bytes": info.st_size,
    })
rows.sort(key=lambda row: row["path"])
tree = hashlib.sha256(json.dumps(
    rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
).encode("ascii")).hexdigest()
if tree != expected_tree:
    raise SystemExit("release exact10 tree digest differs")

def pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError("duplicate manifest key")
        result[key] = value
    return result
def nonfinite(value):
    raise ValueError("nonfinite manifest constant: " + value)
def object_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()
manifest_relative = "release-manifest-v4f.json"
manifest = json.loads(
    raw_by_relative[manifest_relative],
    object_pairs_hook=pairs, parse_constant=nonfinite,
)
unsigned = dict(manifest) if type(manifest) is dict else {}
manifest_digest = unsigned.pop("manifest_digest", None)
payload = manifest.get("payload") if type(manifest) is dict else None
expected_payload = [
    {"relative_path": relative_files[index], "sha256": expected_shas[index]}
    for index in range(9)
]
if (
    manifest.get("schema_version") != manifest_schema
    or manifest.get("status") != manifest_status
    or manifest_digest != expected_manifest_digest
    or object_sha(unsigned) != manifest_digest
    or manifest.get("payload_count") != 9 or type(payload) is not list
    or [
        {"relative_path": row.get("relative_path"), "sha256": row.get("sha256")}
        for row in payload
    ] != expected_payload
    or manifest.get("manifest_target_relative_path") != manifest_relative
    or manifest.get("release_tree_contract", {}).get(
        "exact_file_count_including_manifest") != 10
    or manifest.get("authority_graph", {}).get(
        "sha256_graph_is_directed_and_acyclic") is not True
    or manifest.get("authority_graph", {}).get(
        "runtime_pins_controller_or_manifest") is not False
):
    raise SystemExit("release manifest semantic closure differs")
print("V4F_RELEASE_EXACT10_VALIDATED=" + tree)
PY
    )"; then
    fail "release exact10/manifest validator failed"
  fi
  [[ "${fn_validation}" == "V4F_RELEASE_EXACT10_VALIDATED=${expected_release_tree_sha256}" ]] || \
    fail "release exact10/manifest validator output differs"
}
require_sealed_json() {
  require_plain_file "$1" "$2" 444 "$3"
}

sealed_tree_digest() {
  "$1" -I -S -B - "$2" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys
root = Path(sys.argv[1])
if not root.is_absolute() or root.is_symlink():
    raise SystemExit("tree root differs")
root = root.resolve(strict=True)
rows = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root).as_posix()
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise SystemExit("tree symlink: " + relative)
    if stat.S_ISDIR(info.st_mode):
        if stat.S_IMODE(info.st_mode) != 0o555:
            raise SystemExit("tree directory mode: " + relative)
        continue
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
        raise SystemExit("tree file envelope: " + relative)
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": relative, "sha256": digest.hexdigest(), "size_bytes": info.st_size})
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
print(hashlib.sha256(raw).hexdigest())
PY
}

input_snapshot() {
  local fn_python_bin=$1 fn_feature_root=$2 fn_v4a_receipt=$3
  local fn_v4c_frontier_receipt=$4 fn_v4d_receipt=$5
  "${fn_python_bin}" -I -S -B - \
    "${fn_feature_root}" "${expected_feature_receipt_sha256}" \
    "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" \
    "${fn_v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" \
    "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

feature_root = Path(sys.argv[1])
authorities = [
    (feature_root / "feature_extraction_receipt.json", sys.argv[2], "feature"),
    (Path(sys.argv[3]), sys.argv[4], "v4a"),
    (Path(sys.argv[5]), sys.argv[6], "v4c"),
    (Path(sys.argv[7]), sys.argv[8], "v4d"),
]
rows = []
def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mode,
            value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)
def bind(path, expected, label, parse=False):
    if not path.is_absolute() or path.is_symlink() or str(path) != str(path.resolve(strict=True)):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1:
        raise SystemExit(label + " seal differs")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        closed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if not (identity(before) == identity(opened) == identity(closed) == identity(after)):
        raise SystemExit(label + " single-FD identity differs")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit(label + " SHA differs")
    rows.append({"label": label, "path": str(path), "sha256": expected,
                 "size_bytes": before.st_size})
    return json.loads(raw) if parse else None

feature = None
for path, expected, label in authorities:
    value = bind(path, expected, label, parse=label == "feature")
    if label == "feature":
        feature = value
if not isinstance(feature, dict) or not isinstance(feature.get("shards"), list) or len(feature["shards"]) != 6:
    raise SystemExit("feature exact6 receipt differs")
for index, shard in enumerate(feature["shards"]):
    if not isinstance(shard, dict) or shard.get("index") != index:
        raise SystemExit("feature shard order differs")
    bind(Path(shard["path"]), shard["sha256"], "feature-shard-" + str(index))
raw = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
print(hashlib.sha256(raw).hexdigest())
PY
}

snapshot_authorities() {
  [[ $# -eq 8 ]] || fail "snapshot authority argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_controller_source=$7
  local fn_expected_controller_sha=$8 fn_release_digest fn_input_digest
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python
  require_release "${fn_release_root}" "${fn_python_bin}"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 detached-controller
  require_sealed_json "${fn_feature_root}/feature_extraction_receipt.json" \
    "${expected_feature_receipt_sha256}" feature-receipt
  require_sealed_json "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" v4a-receipt
  require_sealed_json "${fn_v4c_frontier_receipt}" \
    "${expected_v4c_frontier_receipt_sha256}" v4c-frontier-receipt
  require_sealed_json "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" v4d-receipt
  if ! fn_release_digest="$(sealed_tree_digest "${fn_python_bin}" "${fn_release_root}")"; then
    fail "release tree snapshot failed"
  fi
  [[ "${fn_release_digest}" == "${expected_release_tree_sha256}" ]] || \
    fail "release tree SHA differs"
  require_plain_file "${fn_python_bin}" "${expected_python_sha256}" 755 pinned-python-post-snapshot
  if ! fn_input_digest="$(input_snapshot "${fn_python_bin}" "${fn_feature_root}" \
       "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" "${fn_v4d_receipt}")"; then
    fail "input authority snapshot failed"
  fi
  [[ "${fn_input_digest}" =~ ^[0-9a-f]{64}$ ]] || \
    fail "input authority snapshot digest differs"
  printf '%s:%s:%s\n' \
    "${fn_release_digest}" \
    "${fn_input_digest}" \
    "${fn_expected_controller_sha}:${expected_python_sha256}"
}

fresh_cache() {
  local fn_prefix=$1 fn_parent=/tmp fn_cache
  [[ "${fn_parent}" == /* && -d "${fn_parent}" && ! -L "${fn_parent}" && -w "${fn_parent}" ]] || \
    fail "scratch parent differs"
  fn_cache="$(mktemp -d "${fn_parent%/}/${fn_prefix}.XXXXXX")"
  [[ -d "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "fresh cache creation failed"
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
  [[ "${fn_cache}" =~ ^/tmp/${tag}-[A-Za-z0-9-]+\.[A-Za-z0-9]{6}$ \
      || "${fn_cache}" =~ ^/tmp/v4f-miopen-[0-9]+-[0-9]+-[0-9]+-fold[0-4]\.[A-Za-z0-9]{6}$ ]] || \
    fail "unsafe cache cleanup target"
  [[ -d "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup identity differs"
  chmod -R u+w -- "${fn_cache}"
  rm -rf -- "${fn_cache}"
  [[ ! -e "${fn_cache}" && ! -L "${fn_cache}" ]] || fail "cache cleanup failed"
}

gpu_gate() {
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4 fn_fold=$5
  "${fn_python_bin}" -P -B - "${fn_job}" "${fn_node}" "${fn_cache}" "${fn_fold}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, subprocess, sys, time, uuid

job, node, cache_text, fold = sys.argv[1:]
cache = Path(cache_text)
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True))
        or Path("/tmp") not in cache.parents):
    raise SystemExit("fresh node-local cache root differs")
expected_dirs = {"tmp", "xdg", "hf", "pycache", "torch", "user-db",
                 "custom", "triton", "inductor"}
paths = list(cache.rglob("*"))
if ({path.relative_to(cache).as_posix() for path in paths} != expected_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700 for path in paths)):
    raise SystemExit("fresh cache pre-torch exact9 empty-directory closure differs")
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
    raise SystemExit("fresh cache environment binding differs before torch import")

import torch

expected_slurm = {"SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
                  "SLURM_PROCID": "0", "SLURM_LOCALID": "0",
                  "SLURM_CPUS_PER_TASK": "8"}
if any(os.environ.get(key) != value for key, value in expected_slurm.items()):
    raise SystemExit("Slurm exact1 task gate differs")
actual_node = socket.gethostname().split(".", 1)[0]
if actual_node != node or os.environ.get("SLURM_GPUS_ON_NODE") != "1":
    raise SystemExit("Slurm node/GPU count gate differs")
step = os.environ.get("SLURM_STEP_ID", "")
physical_text = os.environ.get("SLURM_STEP_GPUS", "")
if not re.fullmatch(r"[0-9]+", step) or not re.fullmatch(r"[0-9]+", physical_text):
    raise SystemExit("numbered step/physical GPU authority differs")
if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("frozen torch/ROCm runtime differs")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("logical exact1 ROCm gate differs")
properties = torch.cuda.get_device_properties(0)
name = torch.cuda.get_device_name(0)
logical_uuid = str(getattr(properties, "uuid", ""))
if name != "AMD Instinct MI210" or not logical_uuid:
    raise SystemExit("logical MI210 name/UUID gate differs")
try:
    decoded = uuid.UUID(logical_uuid).bytes.decode("ascii").lower()
except (ValueError, UnicodeDecodeError) as error:
    raise SystemExit("logical MI210 UUID does not decode to HSA unique ID") from error
if re.fullmatch(r"[0-9a-f]{16}", decoded) is None:
    raise SystemExit("decoded logical MI210 unique ID differs")
probe_env = os.environ.copy()
for key in ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONNOUSERSITE", "PYTHONPYCACHEPREFIX"):
    probe_env.pop(key, None)
inventory = None
last_output = ""
for _ in range(3):
    probe = subprocess.run(["rocm-smi", "--showuniqueid", "--json"], text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=probe_env)
    last_output = probe.stdout + probe.stderr
    if probe.returncode == 0:
        try:
            candidate = json.loads(probe.stdout)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and len(candidate) == 1:
            inventory = candidate
            break
    time.sleep(0.5)
if inventory is None:
    raise SystemExit("single-device rocm-smi unique-ID probe failed: " + last_output)
card, row = next(iter(inventory.items()))
unique = [str(value) for key, value in row.items() if "unique" in key.lower()]
if len(unique) != 1 or not unique[0]:
    raise SystemExit("physical MI210 UUID gate differs")
def normalized(value):
    value = value.lower().replace("gpu-", "").replace("0x", "")
    return "".join(character for character in value if character in "0123456789abcdef")
if normalized(unique[0]) != decoded:
    raise SystemExit("logical/physical MI210 UUID join differs")
print("V4F_GPU_GATE=" + json.dumps({
    "schema_version": "v4f-vjepa2-exact1-fold-gpu-gate-v1",
    "fold_index": int(fold), "job_id": int(job), "step_id": step,
    "node": actual_node, "ntasks": 1, "nnodes": 1,
    "cpus_per_task": 8,
    "logical_device_count": 1, "visible_gpu_count": 1,
    "slurm_step_gpu_token": int(physical_text),
    "device_name": name, "logical_uuid": logical_uuid,
    "decoded_logical_unique_id": decoded, "rocm_inventory_card": card,
    "physical_uuid": unique[0], "exact_one_uuid_join": True,
    "rocm_smi_unique_only_retry_limit": 3, "torch": str(torch.__version__),
    "torch_hip": str(torch.version.hip), "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
}, sort_keys=True), flush=True)
PY
}

cpu_step_gate() {
  [[ $# -eq 7 ]] || fail "CPU step gate argument count differs"
  local fn_python_bin=$1 fn_job=$2 fn_node=$3 fn_cache=$4 fn_role=$5 fn_schema=$6
  local fn_cpus=$7
  "${fn_python_bin}" -I -S -B - "${fn_job}" "${fn_node}" "${fn_cache}" \
    "${fn_role}" "${fn_schema}" "${fn_cpus}" <<'PY'
from pathlib import Path
import json, os, re, socket, stat, sys

job, node, cache_text, role, schema, cpus = sys.argv[1:]
cache = Path(cache_text)
if (not cache.is_absolute() or cache.is_symlink()
        or str(cache) != str(cache.resolve(strict=True))
        or Path("/tmp") not in cache.parents):
    raise SystemExit("CPU step fresh cache root differs")
expected_dirs = {"tmp", "xdg", "hf", "pycache", "torch", "user-db",
                 "custom", "triton", "inductor"}
paths = list(cache.rglob("*"))
if ({path.relative_to(cache).as_posix() for path in paths} != expected_dirs
        or any(path.is_symlink() or not path.is_dir()
               or stat.S_IMODE(path.lstat().st_mode) != 0o700 for path in paths)):
    raise SystemExit("CPU step fresh cache exact9 closure differs")
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
    raise SystemExit("CPU step cache environment differs")
expected_slurm = {
    "SLURM_JOB_ID": job, "SLURM_NTASKS": "1", "SLURM_NNODES": "1",
    "SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_CPUS_PER_TASK": cpus,
}
if any(os.environ.get(key) != value for key, value in expected_slurm.items()):
    raise SystemExit("CPU exact1 Slurm task gate differs")
step = os.environ.get("SLURM_STEP_ID", "")
if re.fullmatch(r"[0-9]+", step) is None:
    raise SystemExit("CPU step must have a numbered Slurm step ID")
if socket.gethostname().split(".", 1)[0] != node:
    raise SystemExit("CPU step node differs")
if os.environ.get("SLURM_GPUS_ON_NODE") not in (None, "", "0"):
    raise SystemExit("CPU step unexpectedly received a GPU")
if os.environ.get("SLURM_STEP_GPUS") not in (None, ""):
    raise SystemExit("CPU step GPU token differs")
if any(os.environ.get(key) is not None for key in (
    "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)):
    raise SystemExit("CPU step inherited a GPU visibility variable")
print("V4F_CPU_STEP_GATE=" + json.dumps({
    "schema_version": schema, "role": role, "job_id": int(job),
    "step_id": step, "node": node,
    "ntasks": 1, "nnodes": 1, "cpus_per_task": int(cpus),
    "gres": "none", "gpu_visibility_variables_absent": True,
    "fresh_cache_root": str(cache),
    "fresh_cache_pre_torch_exact9_empty_directories": True,
}, sort_keys=True), flush=True)
PY
}

verify_fold_branch() {
  [[ $# -eq 8 ]] || fail "verify fold argument count differs"
  local fn_python_bin=$1 fn_release_root=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_fold_root=$7 fn_fold=$8
  "${fn_python_bin}" -I -S -B - \
    "${fn_fold_root}" "${fn_fold}" "${expected_fold_schema}" \
    "${expected_aggregate_status}" "${expected_inner_no_go_status}" \
    "${expected_checkpoint_schema}" \
    "${fn_release_root}/${runtime_relative}" "${expected_runtime_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/semantic_anchor_vjepa2_analytic_frontier_v4c.py" \
    "${expected_v4c_runtime_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/extract_vjepa2_ordered_contextual_features_v4c.py" \
    "${expected_extractor_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/semantic_anchor_linear_frontier_v4_fast.py" \
    "${expected_v4a_runtime_sha256}" \
    "${fn_release_root}/methods/bernini_action_editing/semantic_anchor_vjepa2_nonlinear_temporal_codec_v4d.py" \
    "${expected_v4d_runtime_sha256}" \
    "${fn_release_root}/${v4e_burned_runtime_relative}" \
    "${expected_v4e_burned_runtime_sha256}" \
    "${fn_feature_root}" "${expected_feature_receipt_sha256}" \
    "${fn_v4a_receipt}" "${expected_v4a_receipt_sha256}" \
    "${fn_v4c_frontier_receipt}" "${expected_v4c_frontier_receipt_sha256}" \
    "${fn_v4d_receipt}" "${expected_v4d_receipt_sha256}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

root = Path(sys.argv[1]); fold_index = int(sys.argv[2])
fold_schema, pass_status, no_go_status, checkpoint_schema = sys.argv[3:7]
implementation = {
    "implementation_path": str(Path(sys.argv[7]).resolve(strict=True)),
    "implementation_sha256": sys.argv[8],
    "v4c_implementation_path": str(Path(sys.argv[9]).resolve(strict=True)),
    "v4c_implementation_sha256": sys.argv[10],
    "extractor_implementation_path": str(Path(sys.argv[11]).resolve(strict=True)),
    "extractor_implementation_sha256": sys.argv[12],
    "v4a_implementation_path": str(Path(sys.argv[13]).resolve(strict=True)),
    "v4a_implementation_sha256": sys.argv[14],
    "v4d_implementation_path": str(Path(sys.argv[15]).resolve(strict=True)),
    "v4d_implementation_sha256": sys.argv[16],
    "v4e_burned_implementation_path": str(Path(sys.argv[17]).resolve(strict=True)),
    "v4e_burned_implementation_sha256": sys.argv[18],
}
feature_root = str(Path(sys.argv[19]).resolve(strict=True)); feature_sha = sys.argv[20]
v4a_path = str(Path(sys.argv[21]).resolve(strict=True)); v4a_sha = sys.argv[22]
v4c_path = str(Path(sys.argv[23]).resolve(strict=True)); v4c_sha = sys.argv[24]
v4d_path = str(Path(sys.argv[25]).resolve(strict=True)); v4d_sha = sys.argv[26]
rho_grid = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0]
oof_counts = [131, 127, 128, 129, 129]
if (
    not root.is_absolute() or root.is_symlink()
    or str(root) != str(root.resolve(strict=True))
    or stat.S_IMODE(root.lstat().st_mode) != 0o555
):
    raise SystemExit("fold root seal differs")

identity = lambda value: (
    value.st_dev, value.st_ino, value.st_size, value.st_mode,
    value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
)
def read_sealed(path, label, expected_mode=0o444, allow_empty=False):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_nlink != 1 or (before.st_size <= 0 and not allow_empty)
    ):
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
    if (
        path.is_symlink()
        or not (identity(before) == identity(opened) == identity(closed) == identity(after))
    ):
        raise SystemExit(label + " same-FD identity differs")
    return b"".join(chunks), before, digest.hexdigest()

def pairs(rows):
    result = {}
    for key, value in rows:
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
def check_artifact(artifact, path, info, file_sha, role, rho):
    return (
        type(artifact) is dict
        and artifact.get("path") == str(path.resolve(strict=True))
        and artifact.get("file_sha256") == file_sha
        and artifact.get("size_bytes") == info.st_size
        and artifact.get("mode_octal") == "0444"
        and artifact.get("nlink") == 1
        and artifact.get("physical_identity") == {
            "device": info.st_dev, "inode": info.st_ino,
            "size_bytes": info.st_size,
        }
        and artifact.get("outer_fold") == fold_index
        and artifact.get("checkpoint_role") == role
        and artifact.get("selected_step") == 1200
        and artifact.get("deployment_rho") == rho
        and artifact.get("implementation_sha256")
            == implementation["implementation_sha256"]
        and artifact.get("single_fd_pre_post_sha256_exact") is True
        and artifact.get("semantic_metadata_state_replay_verified") is True
        and artifact.get("fresh_reload_strict_state_verified") is True
        and artifact.get("fresh_reload_output_bit_exact") is True
        and artifact.get(
            "caller_model_reloaded_from_sealed_artifact_before_next_stage") is True
    )

receipt_raw, receipt_info, receipt_sha = read_sealed(root / "fold.json", "fold receipt")
receipt = json.loads(receipt_raw, object_pairs_hook=pairs, parse_constant=nonfinite)
unsigned = dict(receipt) if type(receipt) is dict else {}
receipt_digest = unsigned.pop("receipt_digest", None)
fold = receipt.get("fold") if type(receipt) is dict else None
training = fold.get("training") if type(fold) is dict else None
selection = fold.get("rho_selection") if type(fold) is dict else None
inner_pass = selection.get("inner_pass") if type(selection) is dict else None
branch = "PASS" if inner_pass is True else "INNER_NO_GO" if inner_pass is False else None
expected_names = (
    {"fold.json", "preselection.pt", "selected.pt"}
    if branch == "PASS" else {"fold.json", "preselection.pt"}
)
if branch is None or {path.name for path in root.iterdir()} != expected_names:
    raise SystemExit("fold branch exact-file closure differs")
pre_raw, pre_info, pre_sha = read_sealed(root / "preselection.pt", "preselection")
selected_info = selected_sha = None
if branch == "PASS":
    _, selected_info, selected_sha = read_sealed(root / "selected.pt", "selected")

pre = fold.get("preselection_checkpoint_artifact") if type(fold) is dict else None
selected = fold.get("selected_checkpoint_artifact") if type(fold) is dict else None
fit_iids = fold.get("model_fit_ordered_iids") if type(fold) is dict else None
inner_iids = fold.get("inner_validation_ordered_iids") if type(fold) is dict else None
oof_iids = fold.get("oof_ordered_iids") if type(fold) is dict else None
evidence = receipt.get("oof_evidence") if type(receipt) is dict else None
candidates = selection.get("candidates") if type(selection) is dict else None
passing = [row["rho"] for row in candidates if type(row) is dict and row.get("pass") is True] \
    if type(candidates) is list else None
selected_rho = fold.get("selected_rho") if type(fold) is dict else None
scope = receipt.get("qualification_scope") if type(receipt) is dict else None
false_scope = {
    "latent_metric_qualified", "action_representation_qualified",
    "identity_disentanglement_qualified", "identity_preservation_qualified",
    "prior_qualified", "prior_generation_qualified", "generation_qualified",
    "renderer_qualified", "video_editing_qualified", "inference_authorized",
    "web_evaluation_authorized", "full644_refit_authorized",
}
stage3 = fold.get("selective_feature_materialization", {}).get(
    "stage3_only_after_selected_checkpoint_strong_seal_reload_or_no_go", {}
) if type(fold) is dict else {}
if (
    type(receipt) is not dict or receipt.get("schema_version") != fold_schema
    or receipt.get("status") != (pass_status if branch == "PASS" else no_go_status)
    or fold.get("fold_status") != receipt.get("status")
    or receipt.get("authority") != "burned_exposed_known_transform_development_fold_only"
    or receipt_digest != object_sha(unsigned)
    or receipt.get("implementation") != implementation
    or receipt.get("fold_root") != str(root)
    or receipt.get("feature_authority", {}).get("feature_root") != feature_root
    or receipt.get("feature_authority", {}).get("feature_receipt_sha256") != feature_sha
    or receipt.get("upstream_authorities", {}).get("v4a_receipt_path") != v4a_path
    or receipt.get("upstream_authorities", {}).get("v4a_receipt_sha256") != v4a_sha
    or receipt.get("upstream_authorities", {}).get("v4c_frontier_receipt_path") != v4c_path
    or receipt.get("upstream_authorities", {}).get("v4c_frontier_receipt_sha256") != v4c_sha
    or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_path") != v4d_path
    or receipt.get("upstream_authorities", {}).get("v4d_burned_receipt_sha256") != v4d_sha
    or type(fold) is not dict or fold.get("fold_index") != fold_index
    or type(training) is not dict
    or training.get("selected_step") != 1200
    or training.get("full_budget_steps_executed") != 1200
    or training.get("checkpoint_steps") != [0, 1200]
    or type(fit_iids) is not list or len(set(fit_iids)) != len(fit_iids)
    or len(fit_iids) != fold.get("model_fit_original_count")
    or object_sha(fit_iids) != fold.get("model_fit_iid_digest")
    or type(inner_iids) is not list or len(set(inner_iids)) != len(inner_iids)
    or len(inner_iids) != fold.get("inner_validation_original_count")
    or object_sha(inner_iids) != fold.get("inner_validation_iid_digest")
    or type(oof_iids) is not list or len(set(oof_iids)) != len(oof_iids)
    or len(oof_iids) != oof_counts[fold_index]
    or object_sha(oof_iids) != fold.get("oof_iid_digest")
    or len(set(fit_iids + inner_iids + oof_iids)) != 644
    or type(evidence) is not list
    or receipt.get("oof_evidence_sha256") != object_sha(evidence)
    or type(candidates) is not list or len(candidates) != 7
    or [row.get("rho") for row in candidates] != rho_grid
    or [row.get("rho_ordinal") for row in candidates] != list(range(7))
    or any(row.get("rho_fp32_exact_power_of_two") is not True for row in candidates)
    or any(type(row.get("bootstrap_seed_ledger")) is not list
           or len(row["bootstrap_seed_ledger"]) != 34 for row in candidates)
    or selection.get("exact7_candidate_ledger_complete") is not True
    or selection.get("rho_grid_preregistered_ascending") != rho_grid
    or selection.get("single_candidate") is not False
    or selection.get("base_state_sha256_before_scan")
        != selection.get("base_state_sha256_after_scan")
    or selection.get("base_state_unchanged_across_all_rho_candidates") is not True
    or selected_rho != (passing[0] if passing else None)
    or not check_artifact(pre, root / "preselection.pt", pre_info, pre_sha,
                          "preselection_fixed_step1200", 1.0)
    or pre.get("preselection_base_state_sha256")
        != training.get("final_step_base_state_sha256")
    or type(scope) is not dict
    or scope.get("exposed_five_view_codec_development_gate") is not None
    or scope.get("aggregate_gate_evaluated") is not False
    or scope.get("vae_necessary") is not None
    or any(scope.get(key) is not False for key in false_scope)
):
    raise SystemExit("fold receipt common semantic closure differs")
if branch == "PASS":
    pair = fold.get("preselection_selected_checkpoint_pair_join")
    if (
        receipt.get("oof_evidence_count") != oof_counts[fold_index]
        or len(evidence) != oof_counts[fold_index]
        or [row.get("iid") for row in evidence] != oof_iids
        or fold.get("oof_semantic_tensor_materialized_count") != 5 * oof_counts[fold_index]
        or selected_rho not in rho_grid or passing == []
        or not check_artifact(selected, root / "selected.pt", selected_info,
                              selected_sha, "selected_fold_local_rho", selected_rho)
        or selected.get("preselection_checkpoint_file_sha256") != pre_sha
        or selected.get("preselection_base_state_sha256")
            != pre.get("preselection_base_state_sha256")
        or type(pair) is not dict
        or pair.get("preselection_device_inode")
            != [pre_info.st_dev, pre_info.st_ino]
        or pair.get("selected_device_inode")
            != [selected_info.st_dev, selected_info.st_ino]
        or [pre_info.st_dev, pre_info.st_ino]
            == [selected_info.st_dev, selected_info.st_ino]
        or pair.get("distinct_device_inode_pair") is not True
        or pair.get("same_preselection_base_state_sha256") is not True
        or fold.get("selected_checkpoint_completed_before_oof_transform_or_model_evaluation")
            is not True
        or stage3.get("semantic_tensor_materialized_count")
            != 5 * oof_counts[fold_index]
    ):
        raise SystemExit("passing fold selected/OOF closure differs")
else:
    if (
        passing != [] or selected_rho is not None or selected is not None
        or (root / "selected.pt").exists() or (root / "selected.pt").is_symlink()
        or fold.get("preselection_selected_checkpoint_pair_join") is not None
        or receipt.get("oof_evidence_count") != 0 or evidence != []
        or fold.get("oof_semantic_tensor_materialized_count") != 0
        or fold.get("oof_semantic_tensor_read_count_exact0_on_inner_no_go") is not True
        or fold.get("selected_checkpoint_completed_before_oof_transform_or_model_evaluation")
            is not False
        or stage3.get("semantic_tensor_materialized_count") != 0
        or stage3.get("oof_semantic_tensor_read_count_exact0") is not True
    ):
        raise SystemExit("INNER_NO_GO exact-zero selected/OOF closure differs")
print(json.dumps({
    "branch": branch, "fold": fold_index,
    "receipt_sha256": receipt_sha, "receipt_digest": receipt_digest,
    "receipt_size_bytes": receipt_info.st_size,
    "preselection_sha256": pre_sha, "preselection_size_bytes": pre_info.st_size,
    "selected_sha256": selected_sha,
    "selected_size_bytes": selected_info.st_size if selected_info is not None else None,
    "selected_step": training["selected_step"], "selected_rho": selected_rho,
    "oof_evidence_count": len(evidence),
}, sort_keys=True, separators=(",", ":")))
PY
}
run_fold_child() {
  [[ $# -eq 12 ]] || fail "internal fold child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_fold_root=$7 fn_fold=$8
  local fn_job=$9 fn_node=${10} fn_expected_controller_sha=${11} fn_pre_snapshot=${12}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_fold_parent
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${worker_folds[${fn_fold}]}" == "${fn_fold}" \
      && "${worker_jobs[${fn_fold}]}" == "${fn_job}" \
      && "${worker_nodes[${fn_fold}]}" == "${fn_node}" ]] || \
    fail "fold holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 detached-controller-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "child allocation binding differs"
  [[ "${fn_fold_root}" == /* && ! -e "${fn_fold_root}" && ! -L "${fn_fold_root}" ]] || \
    fail "fold root is not fresh"
  fn_fold_parent="$(dirname -- "${fn_fold_root}")"
  [[ "$(basename -- "${fn_fold_root}")" == "fold${fn_fold}" \
      && -d "${fn_fold_parent}" && ! -L "${fn_fold_parent}" \
      && "${fn_fold_parent}" == "$(readlink -f -- "${fn_fold_parent}")" \
      && "$(stat -c %a "${fn_fold_parent}")" == 700 ]] || \
    fail "fold root parent/name authority differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "fold child pre-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "fold child pre-run authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "v4f-miopen-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}-fold${fn_fold}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  gpu_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" "${fn_fold}"
  mkdir "${fn_fold_root}"
  chmod 0700 "${fn_fold_root}"
  "${fn_python_bin}" -P -B "${fn_runtime}" train-fold \
    --feature-root "${fn_feature_root}" \
    --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
    --v4a-receipt "${fn_v4a_receipt}" \
    --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
    --v4c-frontier-receipt "${fn_v4c_frontier_receipt}" \
    --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
    --v4d-receipt "${fn_v4d_receipt}" \
    --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
    --fold-index "${fn_fold}" --fold-root "${fn_fold_root}" --device cuda:0
  chmod 0555 "${fn_fold_root}"
  verify_fold_branch "${fn_python_bin}" "${fn_release_root}" "${fn_feature_root}" \
    "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" "${fn_v4d_receipt}" \
    "${fn_fold_root}" "${fn_fold}"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "fold child post-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "code/input authority changed during fold child"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4F_CACHE_CLEANED={"fold_index":%s,"fresh_cache_root":"%s","absent_after_child_cleanup":true}\n' \
    "${fn_fold}" "${fn_cache}"
}

run_exact_test_suite() {
  [[ $# -eq 3 ]] || fail "exact test runner argument count differs"
  local fn_python_bin=$1 fn_optimized=$2 fn_test_module=$3
  local -a fn_flags=(-P -B)
  if [[ "${fn_optimized}" == true ]]; then
    fn_flags=(-O -P -B)
  elif [[ "${fn_optimized}" != false ]]; then
    fail "exact test runner optimization flag differs"
  fi
  "${fn_python_bin}" "${fn_flags[@]}" - "${fn_test_module}" <<'PY'
import importlib, json, sys, unittest
import torch

if str(torch.__version__) != "2.7.1+rocm6.3" or not torch.version.hip:
    raise SystemExit("preflight frozen torch/ROCm runtime differs")
module = importlib.import_module(sys.argv[1])
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
summary = {
    "errors": len(result.errors), "expected_failures": len(result.expectedFailures),
    "failures": len(result.failures), "optimized": bool(sys.flags.optimize),
    "skipped": len(result.skipped), "tests_run": result.testsRun,
    "unexpected_successes": len(result.unexpectedSuccesses),
}
if summary != {
    "errors": 0, "expected_failures": 0, "failures": 0,
    "optimized": bool(sys.flags.optimize), "skipped": 0, "tests_run": 30,
    "unexpected_successes": 0,
}:
    raise SystemExit("preflight exact30 no-skip test closure differs: " + repr(summary))
print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
PY
}

run_preflight_child() {
  [[ $# -eq 10 ]] || fail "internal preflight child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_job=$7 fn_node=$8
  local fn_expected_controller_sha=$9 fn_pre_snapshot=${10}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_gate fn_gate_json
  local fn_train_help fn_aggregate_help fn_token fn_test_module
  local fn_normal_stdout fn_normal_stderr fn_optimized_stdout fn_optimized_stderr
  local fn_normal_pid fn_optimized_pid fn_normal_rc fn_optimized_rc
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${preflight_job}" && "${fn_node}" == "${preflight_node}" ]] || \
    fail "preflight fixed holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 \
    detached-controller-preflight-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "preflight allocation binding differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "preflight child initial authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "preflight child initial authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_cache="$(fresh_cache "${tag}-preflight-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  fn_gate="$(cpu_step_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    preflight v4f-vjepa2-exact1-cpu-preflight-gate-v1 "${preflight_cpus}")"
  [[ "${fn_gate}" == V4F_CPU_STEP_GATE=* ]] || fail "preflight CPU gate output differs"
  fn_gate_json=${fn_gate#V4F_CPU_STEP_GATE=}
  fn_test_module=methods.bernini_action_editing.tests.test_semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy
  fn_normal_stdout="${fn_cache}/normal-tests.stdout"
  fn_normal_stderr="${fn_cache}/normal-tests.stderr"
  fn_optimized_stdout="${fn_cache}/optimized-tests.stdout"
  fn_optimized_stderr="${fn_cache}/optimized-tests.stderr"
  (
    set -o noclobber
    cd "${fn_release_root}"
    run_exact_test_suite "${fn_python_bin}" false "${fn_test_module}" \
      >"${fn_normal_stdout}" 2>"${fn_normal_stderr}"
  ) &
  fn_normal_pid=$!
  (
    set -o noclobber
    cd "${fn_release_root}"
    run_exact_test_suite "${fn_python_bin}" true "${fn_test_module}" \
      >"${fn_optimized_stdout}" 2>"${fn_optimized_stderr}"
  ) &
  fn_optimized_pid=$!
  fn_normal_rc=0
  fn_optimized_rc=0
  wait "${fn_normal_pid}" || fn_normal_rc=$?
  wait "${fn_optimized_pid}" || fn_optimized_rc=$?
  if (( fn_normal_rc != 0 || fn_optimized_rc != 0 )); then
    printf '%s\n' "normal test rc=${fn_normal_rc}" "optimized test rc=${fn_optimized_rc}" >&2
    sed -n '1,240p' "${fn_normal_stdout}" "${fn_normal_stderr}" \
      "${fn_optimized_stdout}" "${fn_optimized_stderr}" >&2
    fail "parallel exact30 no-skip test suites failed"
  fi
  "${fn_python_bin}" -I -S -B - "${fn_normal_stdout}" "${fn_optimized_stdout}" \
    "${fn_normal_stderr}" "${fn_optimized_stderr}" <<'PY'
from pathlib import Path
import json, sys
expected_common = {
    "errors": 0, "expected_failures": 0, "failures": 0, "skipped": 0,
    "tests_run": 30, "unexpected_successes": 0,
}
for index, optimized in enumerate((False, True), start=1):
    value = json.loads(Path(sys.argv[index]).read_text("ascii"))
    expected = {**expected_common, "optimized": optimized}
    if value != expected:
        raise SystemExit("parallel test result JSON differs")
for index in (3, 4):
    lines = Path(sys.argv[index]).read_text(encoding="utf-8", errors="strict").splitlines()
    if (not any(line.startswith("Ran 30 tests in ") for line in lines)
            or "OK" not in lines
            or any("skipped=" in line for line in lines)):
        raise SystemExit("parallel unittest text summary differs")
PY
  (
    cd "${fn_release_root}"
    "${fn_python_bin}" -P -B -m py_compile "${release_relative_files[@]}"
  )
  fn_train_help="$("${fn_python_bin}" -P -B "${fn_runtime}" train-fold --help)"
  fn_aggregate_help="$("${fn_python_bin}" -P -B "${fn_runtime}" aggregate --help)"
  for fn_token in --feature-root --expected-feature-receipt-sha256 \
    --v4a-receipt --expected-v4a-receipt-sha256 \
    --v4c-frontier-receipt --expected-v4c-frontier-receipt-sha256 \
    --v4d-receipt --expected-v4d-receipt-sha256 --fold-root; do
    grep -F -- "${fn_token}" <<<"${fn_train_help}" >/dev/null || \
      fail "train-fold help misses ${fn_token}"
    grep -F -- "${fn_token}" <<<"${fn_aggregate_help}" >/dev/null || \
      fail "aggregate help misses ${fn_token}"
  done
  for fn_token in --fold-index --device; do
    grep -F -- "${fn_token}" <<<"${fn_train_help}" >/dev/null || \
      fail "train-fold help misses ${fn_token}"
  done
  grep -F -- --output <<<"${fn_aggregate_help}" >/dev/null || \
    fail "aggregate help misses --output"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "preflight child post-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "code/input authority changed during CPU preflight"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4F_PREFLIGHT_PASS={"job":%s,"node":"%s","cpus":%s,"memory":"4G","gres":"none","cpu_gate":%s,"normal_tests_passed":true,"normal_tests_run":30,"normal_tests_skipped":0,"optimized_tests_passed":true,"optimized_tests_run":30,"optimized_tests_skipped":0,"tests_ran_in_parallel":true,"omp_threads_per_test_process":1,"compile_passed":true,"cli_help_passed":true,"fresh_cache_cleaned":true}\n' \
    "${fn_job}" "${fn_node}" "${preflight_cpus}" "${fn_gate_json}"
}

run_aggregate_child() {
  [[ $# -eq 11 ]] || fail "internal aggregate child argument count differs"
  local fn_release_root=$1 fn_python_bin=$2 fn_feature_root=$3 fn_v4a_receipt=$4
  local fn_v4c_frontier_receipt=$5 fn_v4d_receipt=$6 fn_run_root=$7 fn_job=$8
  local fn_node=$9 fn_expected_controller_sha=${10} fn_pre_snapshot=${11}
  local fn_controller_source fn_runtime fn_cache fn_post_snapshot fn_output
  local fn_aggregate_members
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${aggregate_job}" && "${fn_node}" == "${aggregate_node}" ]] || \
    fail "aggregate fixed holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 \
    detached-controller-aggregate-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "aggregate allocation binding differs"
  [[ "${fn_run_root}" == /* && -d "${fn_run_root}" && ! -L "${fn_run_root}" \
      && "${fn_run_root}" == "$(readlink -f -- "${fn_run_root}")" \
      && "$(stat -c %a "${fn_run_root}")" == 700 ]] || \
    fail "aggregate run root differs"
  if ! fn_aggregate_members="$(find "${fn_run_root}/aggregate" -mindepth 1 -print -quit)"; then
    fail "aggregate fresh-directory scan failed"
  fi
  [[ -d "${fn_run_root}/aggregate" && ! -L "${fn_run_root}/aggregate" \
      && "$(stat -c %a "${fn_run_root}/aggregate")" == 700 \
      && -z "${fn_aggregate_members}" ]] || \
    fail "aggregate output directory is not fresh exact0"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "aggregate child pre-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "aggregate child pre-run authority differs"
  fn_runtime="${fn_release_root}/${runtime_relative}"
  fn_output="${fn_run_root}/aggregate/receipt.json"
  fn_cache="$(fresh_cache "${tag}-aggregate-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  export PYTHONPATH="${fn_release_root}" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONSAFEPATH=1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  cpu_step_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    aggregate v4f-vjepa2-exact1-cpu-aggregate-gate-v1 "${aggregate_cpus}"
  "${fn_python_bin}" -P -B "${fn_runtime}" aggregate \
    --feature-root "${fn_feature_root}" \
    --expected-feature-receipt-sha256 "${expected_feature_receipt_sha256}" \
    --v4a-receipt "${fn_v4a_receipt}" \
    --expected-v4a-receipt-sha256 "${expected_v4a_receipt_sha256}" \
    --v4c-frontier-receipt "${fn_v4c_frontier_receipt}" \
    --expected-v4c-frontier-receipt-sha256 "${expected_v4c_frontier_receipt_sha256}" \
    --v4d-receipt "${fn_v4d_receipt}" \
    --expected-v4d-receipt-sha256 "${expected_v4d_receipt_sha256}" \
    --fold-root "${fn_run_root}/fold0" --fold-root "${fn_run_root}/fold1" \
    --fold-root "${fn_run_root}/fold2" --fold-root "${fn_run_root}/fold3" \
    --fold-root "${fn_run_root}/fold4" --output "${fn_output}"
  [[ "$(stat -c '%a:%h' "${fn_output}")" == 444:1 && ! -L "${fn_output}" ]] || \
    fail "aggregate receipt output seal differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "aggregate child post-run authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "code/input authority changed during aggregate child"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  printf 'V4F_CACHE_CLEANED={"role":"aggregate","fresh_cache_root":"%s","absent_after_child_cleanup":true}\n' \
    "${fn_cache}"
}

run_branch_seal_child() {
  [[ $# -eq 12 ]] || fail "internal branch seal child argument count differs"
  local fn_mode=$1 fn_release_root=$2 fn_python_bin=$3 fn_feature_root=$4
  local fn_v4a_receipt=$5 fn_v4c_frontier_receipt=$6 fn_v4d_receipt=$7
  local fn_run_root=$8 fn_job=$9 fn_node=${10}
  local fn_expected_controller_sha=${11} fn_pre_snapshot=${12}
  local fn_controller_source fn_cache fn_gate fn_gate_json fn_result fn_post_snapshot
  [[ "${fn_mode}" == success || "${fn_mode}" == inner-no-go ]] || \
    fail "branch seal mode differs"
  fn_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"
  [[ "${fn_job}" == "${aggregate_job}" && "${fn_node}" == "${aggregate_node}" ]] || \
    fail "branch postflight fixed holder mapping differs"
  require_plain_file "${fn_controller_source}" "${fn_expected_controller_sha}" 555 \
    detached-controller-branch-postflight-child
  [[ "${fn_controller_source}" != "${fn_release_root%/}/"* ]] || \
    fail "controller must remain detached from release"
  [[ "${SLURM_JOB_ID:-}" == "${fn_job}" ]] || fail "postflight allocation binding differs"
  [[ "${fn_run_root}" == /* && -d "${fn_run_root}" && ! -L "${fn_run_root}" \
      && "${fn_run_root}" == "$(readlink -f -- "${fn_run_root}")" \
      && "$(stat -c %a "${fn_run_root}")" == 700 \
      && ! -e "${fn_run_root}/seal.json" && ! -L "${fn_run_root}/seal.json" \
      && ! -e "${fn_run_root}/inner-no-go-seal.json" \
      && ! -L "${fn_run_root}/inner-no-go-seal.json" ]] || \
    fail "branch postflight run root is not fresh for seal"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "branch postflight initial authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "branch postflight initial authority differs"
  fn_cache="$(fresh_cache "${tag}-postflight-${fn_mode}-${fn_job}-${SLURM_STEP_ID:?}-${SLURM_PROCID:?}")"
  trap 'cleanup_cache "${fn_cache}"' EXIT
  activate_cache "${fn_cache}"
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
  fn_gate="$(cpu_step_gate "${fn_python_bin}" "${fn_job}" "${fn_node}" "${fn_cache}" \
    postflight "v4f-vjepa2-exact1-cpu-${fn_mode}-postflight-gate-v1" "${aggregate_cpus}")"
  [[ "${fn_gate}" == V4F_CPU_STEP_GATE=* ]] || fail "branch CPU gate output differs"
  fn_gate_json=${fn_gate#V4F_CPU_STEP_GATE=}
  fn_result="$("${fn_python_bin}" -I -S -B - \
    "${fn_mode}" "${fn_run_root}" "${fn_release_root}" "${fn_python_bin}" \
    "${fn_controller_source}" "${fn_expected_controller_sha}" "${fn_pre_snapshot}" \
    "${fn_gate_json}" "${expected_release_tree_sha256}" \
    "${expected_release_manifest_sha256}" "${expected_release_manifest_digest}" \
    "${expected_runtime_sha256}" "${expected_runtime_test_sha256}" \
    "${expected_python_sha256}" "${expected_fold_schema}" \
    "${expected_aggregate_schema}" "${expected_aggregate_status}" \
    "${expected_inner_no_go_status}" "${expected_checkpoint_schema}" <<'PY'
from pathlib import Path
import hashlib, json, os, stat, sys

(mode, root_s, release_s, python_s, controller_s, controller_sha,
 authority_snapshot, cpu_gate_s, release_tree_sha, release_manifest_sha,
 release_manifest_digest, runtime_sha, tests_sha, python_sha, fold_schema,
 aggregate_schema, pass_status, no_go_status, checkpoint_schema) = sys.argv[1:]
root = Path(root_s); release = Path(release_s); controller = Path(controller_s)
runtime_path = release / "methods/bernini_action_editing/semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py"
tests_path = release / "methods/bernini_action_editing/tests/test_semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy.py"
manifest_path = release / "release-manifest-v4f.json"
seal_name = "seal.json" if mode == "success" else "inner-no-go-seal.json"
seal_path = root / seal_name
oof_counts = [131, 127, 128, 129, 129]
rho_grid = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0]
worker_jobs = [143808, 143812, 143811, 143808, 143808]
worker_nodes = [
    "auh7-1b-gpu-315", "auh7-1b-gpu-293", "auh7-1b-gpu-306",
    "auh7-1b-gpu-233", "auh7-1b-gpu-268",
]
if mode not in {"success", "inner-no-go"}:
    raise SystemExit("postflight branch differs")
if (
    not root.is_absolute() or root.is_symlink()
    or str(root) != str(root.resolve(strict=True))
    or stat.S_IMODE(root.lstat().st_mode) != 0o700
    or seal_path.exists() or seal_path.is_symlink()
):
    raise SystemExit("postflight root envelope differs")

def pairs(rows):
    result = {}
    for key, value in rows:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result
def nonfinite(value):
    raise ValueError("nonfinite JSON constant: " + value)
def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()
def identity(value):
    return (
        value.st_dev, value.st_ino, value.st_size, value.st_mode,
        value.st_nlink, value.st_mtime_ns, value.st_ctime_ns,
    )
def read_sealed(path, label, expected_mode=0o444, allow_empty=False):
    if path.is_symlink() or not path.is_file() or not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit(label + " path differs")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != expected_mode
        or before.st_nlink != 1 or (before.st_size <= 0 and not allow_empty)
    ):
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
    if (
        path.is_symlink()
        or not (identity(before) == identity(opened) == identity(closed) == identity(after))
    ):
        raise SystemExit(label + " same-FD identity differs")
    return b"".join(chunks), before, digest.hexdigest()
def read_json(path, label, digest_key=None):
    raw, info, file_sha = read_sealed(path, label)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    if type(value) is not dict:
        raise SystemExit(label + " root differs")
    if digest_key is not None:
        unsigned = dict(value); digest = unsigned.pop(digest_key, None)
        if digest != object_sha(unsigned):
            raise SystemExit(label + " selfdigest differs")
    else:
        digest = None
    return value, info, file_sha, digest
def artifact_matches(artifact, path, info, file_sha, role, fold, rho):
    return (
        type(artifact) is dict
        and artifact.get("path") == str(path.resolve(strict=True))
        and artifact.get("file_sha256") == file_sha
        and artifact.get("size_bytes") == info.st_size
        and artifact.get("mode_octal") == "0444" and artifact.get("nlink") == 1
        and artifact.get("physical_identity") == {
            "device": info.st_dev, "inode": info.st_ino,
            "size_bytes": info.st_size,
        }
        and artifact.get("checkpoint_role") == role
        and artifact.get("outer_fold") == fold
        and artifact.get("selected_step") == 1200
        and artifact.get("deployment_rho") == rho
        and artifact.get("implementation_sha256") == runtime_sha
        and artifact.get("single_fd_pre_post_sha256_exact") is True
        and artifact.get("semantic_metadata_state_replay_verified") is True
        and artifact.get("fresh_reload_strict_state_verified") is True
        and artifact.get("fresh_reload_output_bit_exact") is True
        and artifact.get(
            "caller_model_reloaded_from_sealed_artifact_before_next_stage") is True
    )
def parse_prefixed(line, prefix, label):
    if not line.startswith(prefix):
        raise SystemExit(label + " prefix differs")
    return json.loads(line[len(prefix):], object_pairs_hook=pairs, parse_constant=nonfinite)
def check_false_scope(scope, label):
    keys = {
        "latent_metric_qualified", "action_representation_qualified",
        "identity_disentanglement_qualified", "identity_preservation_qualified",
        "prior_qualified", "prior_generation_qualified", "generation_qualified",
        "renderer_qualified", "video_editing_qualified", "inference_authorized",
        "web_evaluation_authorized", "full644_refit_authorized",
    }
    if type(scope) is not dict or any(scope.get(key) is not False for key in keys):
        raise SystemExit(label + " qualification scope differs")

source_bindings = {}
for path, expected, label, expected_mode in (
    (manifest_path, release_manifest_sha, "release manifest", 0o444),
    (runtime_path, runtime_sha, "runtime", 0o444),
    (tests_path, tests_sha, "runtime tests", 0o444),
    (Path(python_s), python_sha, "python", 0o755),
    (controller, controller_sha, "detached controller", 0o555),
):
    _, info, actual = read_sealed(path, label, expected_mode)
    if actual != expected:
        raise SystemExit(label + " SHA differs")
    source_bindings[label] = {
        "path": str(path.resolve(strict=True)), "sha256": actual,
        "size_bytes": info.st_size, "mode_octal": format(expected_mode, "04o"),
        "nlink": info.st_nlink,
        "physical_identity": [info.st_dev, info.st_ino],
    }
manifest, _, _, manifest_self = read_json(
    manifest_path, "release manifest", "manifest_digest"
)
if (
    manifest_self != release_manifest_digest
    or manifest.get("schema_version")
        != "v4f-vjepa2-residual-homotopy-detached-release-manifest-v1"
    or manifest.get("status") != "V4F_DETACHED_RELEASE_MANIFEST_SEALED"
    or manifest.get("payload_count") != 9
    or manifest.get("authority_graph", {}).get(
        "runtime_pins_controller_or_manifest") is not False
):
    raise SystemExit("release manifest detached DAG differs")
cpu_gate = json.loads(cpu_gate_s, object_pairs_hook=pairs, parse_constant=nonfinite)
if (
    cpu_gate.get("schema_version")
        != f"v4f-vjepa2-exact1-cpu-{mode}-postflight-gate-v1"
    or cpu_gate.get("role") != "postflight"
    or cpu_gate.get("job_id") != 143811
    or cpu_gate.get("node") != "auh7-1b-gpu-306"
    or cpu_gate.get("gres") != "none"
    or cpu_gate.get("gpu_visibility_variables_absent") is not True
):
    raise SystemExit("postflight CPU gate differs")

launch, _, launch_sha, _ = read_json(root / "launch-plan.json", "launch plan")
preflight = launch.get("cpu_preflight") if type(launch) is dict else None
if (
    launch.get("schema_version") != "v4f-exact5-parallel-launch-plan-v1"
    or launch.get("controller_sha256") != controller_sha
    or launch.get("authority_snapshot") != authority_snapshot
    or launch.get("source_authority", {}).get("release_tree_sha256") != release_tree_sha
    or launch.get("source_authority", {}).get("release_manifest_sha256")
        != release_manifest_sha
    or launch.get("source_authority", {}).get("release_manifest_digest")
        != release_manifest_digest
    or launch.get("source_authority", {}).get("runtime_sha256") != runtime_sha
    or launch.get("source_authority", {}).get("runtime_test_sha256") != tests_sha
    or launch.get("source_authority", {}).get("python_sha256") != python_sha
    or launch.get("receipt_contract", {}).get("fold_schema") != fold_schema
    or launch.get("receipt_contract", {}).get("aggregate_schema") != aggregate_schema
    or launch.get("receipt_contract", {}).get("pass_status") != pass_status
    or launch.get("receipt_contract", {}).get("inner_no_go_status") != no_go_status
    or launch.get("receipt_contract", {}).get("checkpoint_schema") != checkpoint_schema
    or launch.get("workers") != [
        {"fold": index, "job": worker_jobs[index], "node": worker_nodes[index]}
        for index in range(5)
    ]
    or type(preflight) is not dict
    or preflight.get("cpu_gate", {}).get("schema_version")
        != "v4f-vjepa2-exact1-cpu-preflight-gate-v1"
    or preflight.get("job") != 143811
    or preflight.get("node") != "auh7-1b-gpu-306"
    or preflight.get("normal_tests_passed") is not True
    or preflight.get("normal_tests_run") != 30
    or preflight.get("normal_tests_skipped") != 0
    or preflight.get("optimized_tests_passed") is not True
    or preflight.get("optimized_tests_run") != 30
    or preflight.get("optimized_tests_skipped") != 0
    or preflight.get("compile_passed") is not True
    or preflight.get("cli_help_passed") is not True
    or preflight.get("fresh_cache_cleaned") is not True
):
    raise SystemExit("launch plan semantic closure differs")

fold_summary = []; preseal_expected = {"launch-plan.json"}
passing_count = 0; total_oof_count = 0
for fold_index in range(5):
    stdout_rel = f"logs/fold{fold_index}.stdout"
    stderr_rel = f"logs/fold{fold_index}.stderr"
    fold_rel = f"fold{fold_index}/fold.json"
    pre_rel = f"fold{fold_index}/preselection.pt"
    preseal_expected.update({stdout_rel, stderr_rel, fold_rel, pre_rel})
    stdout_raw, _, _, = read_sealed(root / stdout_rel, f"fold{fold_index} stdout")
    stderr_raw, _, _ = read_sealed(
        root / stderr_rel, f"fold{fold_index} stderr", allow_empty=True
    )
    if stderr_raw != b"":
        raise SystemExit(f"fold{fold_index} stderr is not exact0")
    lines = stdout_raw.decode("utf-8").splitlines()
    if len(lines) != 4:
        raise SystemExit(f"fold{fold_index} stdout line count differs")
    gate = parse_prefixed(lines[0], "V4F_GPU_GATE=", "fold GPU gate")
    runtime_result = json.loads(lines[1], object_pairs_hook=pairs, parse_constant=nonfinite)
    verifier = json.loads(lines[2], object_pairs_hook=pairs, parse_constant=nonfinite)
    cleanup = parse_prefixed(lines[3], "V4F_CACHE_CLEANED=", "fold cache")
    if (
        gate.get("schema_version") != "v4f-vjepa2-exact1-fold-gpu-gate-v1"
        or gate.get("fold_index") != fold_index
        or gate.get("job_id") != worker_jobs[fold_index]
        or gate.get("node") != worker_nodes[fold_index]
        or gate.get("torch") != "2.7.1+rocm6.3"
        or gate.get("visible_gpu_count") != 1
        or gate.get("exact_one_uuid_join") is not True
        or cleanup.get("fold_index") != fold_index
        or cleanup.get("absent_after_child_cleanup") is not True
    ):
        raise SystemExit(f"fold{fold_index} GPU/cache gate differs")
    receipt, receipt_info, receipt_sha, receipt_digest = read_json(
        root / fold_rel, f"fold{fold_index} receipt", "receipt_digest"
    )
    fold = receipt.get("fold"); selection = fold.get("rho_selection") if type(fold) is dict else None
    inner_pass = selection.get("inner_pass") if type(selection) is dict else None
    branch = "PASS" if inner_pass is True else "INNER_NO_GO" if inner_pass is False else None
    if (
        branch is None or receipt.get("schema_version") != fold_schema
        or receipt.get("status") != (pass_status if branch == "PASS" else no_go_status)
        or receipt.get("implementation", {}).get("implementation_sha256") != runtime_sha
        or type(fold) is not dict or fold.get("fold_index") != fold_index
        or fold.get("fold_status") != receipt.get("status")
        or fold.get("training", {}).get("selected_step") != 1200
        or fold.get("training", {}).get("full_budget_steps_executed") != 1200
        or selection.get("rho_grid_preregistered_ascending") != rho_grid
        or selection.get("rho_candidate_count") != 7
        or selection.get("single_candidate") is not False
        or selection.get("exact7_candidate_ledger_complete") is not True
        or len(selection.get("candidates", [])) != 7
        or [row.get("rho") for row in selection["candidates"]] != rho_grid
        or [row.get("rho_ordinal") for row in selection["candidates"]] != list(range(7))
        or selection.get("base_state_sha256_before_scan")
            != selection.get("base_state_sha256_after_scan")
        or selection.get("base_state_unchanged_across_all_rho_candidates") is not True
    ):
        raise SystemExit(f"fold{fold_index} nested contract differs")
    passing_rhos = [
        row["rho"] for row in selection["candidates"] if row.get("pass") is True
    ]
    selected_rho = fold.get("selected_rho")
    if selected_rho != (passing_rhos[0] if passing_rhos else None):
        raise SystemExit(f"fold{fold_index} first-pass rho differs")
    pre_raw, pre_info, pre_sha = read_sealed(root / pre_rel, f"fold{fold_index} preselection")
    pre = fold.get("preselection_checkpoint_artifact")
    if not artifact_matches(
        pre, root / pre_rel, pre_info, pre_sha,
        "preselection_fixed_step1200", fold_index, 1.0,
    ):
        raise SystemExit(f"fold{fold_index} preselection artifact differs")
    evidence = receipt.get("oof_evidence")
    check_false_scope(receipt.get("qualification_scope"), f"fold{fold_index}")
    if branch == "PASS":
        selected_rel = f"fold{fold_index}/selected.pt"
        preseal_expected.add(selected_rel); passing_count += 1
        _, selected_info, selected_sha = read_sealed(
            root / selected_rel, f"fold{fold_index} selected"
        )
        selected = fold.get("selected_checkpoint_artifact")
        pair = fold.get("preselection_selected_checkpoint_pair_join")
        if (
            not artifact_matches(
                selected, root / selected_rel, selected_info, selected_sha,
                "selected_fold_local_rho", fold_index, selected_rho,
            )
            or selected.get("preselection_checkpoint_file_sha256") != pre_sha
            or selected.get("preselection_base_state_sha256")
                != pre.get("preselection_base_state_sha256")
            or type(pair) is not dict
            or pair.get("preselection_device_inode")
                != [pre_info.st_dev, pre_info.st_ino]
            or pair.get("selected_device_inode")
                != [selected_info.st_dev, selected_info.st_ino]
            or pair.get("distinct_device_inode_pair") is not True
            or [pre_info.st_dev, pre_info.st_ino]
                == [selected_info.st_dev, selected_info.st_ino]
            or type(evidence) is not list or len(evidence) != oof_counts[fold_index]
            or receipt.get("oof_evidence_count") != oof_counts[fold_index]
            or fold.get("oof_semantic_tensor_materialized_count")
                != 5 * oof_counts[fold_index]
        ):
            raise SystemExit(f"fold{fold_index} PASS artifact/OOF closure differs")
        total_oof_count += len(evidence)
    else:
        if (
            passing_rhos != [] or selected_rho is not None
            or fold.get("selected_checkpoint_artifact") is not None
            or fold.get("preselection_selected_checkpoint_pair_join") is not None
            or (root / f"fold{fold_index}/selected.pt").exists()
            or (root / f"fold{fold_index}/selected.pt").is_symlink()
            or evidence != [] or receipt.get("oof_evidence_count") != 0
            or fold.get("oof_semantic_tensor_materialized_count") != 0
            or fold.get("oof_semantic_tensor_read_count_exact0_on_inner_no_go") is not True
        ):
            raise SystemExit(f"fold{fold_index} INNER_NO_GO closure differs")
    if (
        runtime_result.get("fold") != fold_index
        or runtime_result.get("status") != receipt.get("status")
        or runtime_result.get("fold_receipt_sha256") != receipt_sha
        or runtime_result.get("fold_receipt_digest") != receipt_digest
        or runtime_result.get("selected_step") != 1200
        or runtime_result.get("selected_rho") != selected_rho
        or runtime_result.get("oof_semantic_tensor_materialized_count")
            != fold.get("oof_semantic_tensor_materialized_count")
        or verifier.get("fold") != fold_index or verifier.get("branch") != branch
        or verifier.get("receipt_sha256") != receipt_sha
        or verifier.get("receipt_digest") != receipt_digest
        or verifier.get("preselection_sha256") != pre_sha
        or verifier.get("selected_rho") != selected_rho
        or verifier.get("oof_evidence_count") != len(evidence)
    ):
        raise SystemExit(f"fold{fold_index} stdout/receipt join differs")
    fold_summary.append({
        "fold": fold_index, "branch": branch,
        "receipt_sha256": receipt_sha, "receipt_digest": receipt_digest,
        "preselection_sha256": pre_sha,
        "selected_sha256": selected_sha if branch == "PASS" else None,
        "selected_step": 1200, "selected_rho": selected_rho,
        "oof_evidence_count": len(evidence),
    })

aggregate_binding = None
if mode == "success":
    if passing_count != 5 or total_oof_count != 644:
        raise SystemExit("success branch is not exact5 PASS/exact644")
    preseal_expected.update({
        "logs/aggregate.stdout", "logs/aggregate.stderr", "aggregate/receipt.json",
    })
    aggregate_stdout, _, _ = read_sealed(
        root / "logs/aggregate.stdout", "aggregate stdout"
    )
    aggregate_stderr, _, _ = read_sealed(
        root / "logs/aggregate.stderr", "aggregate stderr", allow_empty=True
    )
    if aggregate_stderr != b"":
        raise SystemExit("aggregate stderr is not exact0")
    lines = aggregate_stdout.decode("utf-8").splitlines()
    if len(lines) != 3:
        raise SystemExit("aggregate stdout line count differs")
    aggregate_gate = parse_prefixed(lines[0], "V4F_CPU_STEP_GATE=", "aggregate CPU gate")
    aggregate_result = json.loads(lines[1], object_pairs_hook=pairs, parse_constant=nonfinite)
    aggregate_cleanup = parse_prefixed(
        lines[2], "V4F_CACHE_CLEANED=", "aggregate cache"
    )
    receipt, info, receipt_sha, receipt_digest = read_json(
        root / "aggregate/receipt.json", "aggregate receipt", "receipt_digest"
    )
    metrics = receipt.get("metrics")
    folds = receipt.get("folds")
    check_false_scope(receipt.get("qualification_scope"), "aggregate")
    if (
        aggregate_gate.get("schema_version")
            != "v4f-vjepa2-exact1-cpu-aggregate-gate-v1"
        or aggregate_gate.get("role") != "aggregate"
        or aggregate_gate.get("job_id") != 143811
        or aggregate_gate.get("node") != "auh7-1b-gpu-306"
        or aggregate_cleanup.get("role") != "aggregate"
        or aggregate_cleanup.get("absent_after_child_cleanup") is not True
        or receipt.get("schema_version") != aggregate_schema
        or receipt.get("status") != pass_status
        or receipt.get("implementation", {}).get("implementation_sha256") != runtime_sha
        or type(folds) is not list or len(folds) != 5
        or [fold["fold_index"] for fold in folds] != list(range(5))
        or receipt.get("training_contract", {}).get("selected_steps_by_fold")
            != [1200] * 5
        or receipt.get("training_contract", {}).get("selected_rhos_by_fold")
            != [row["selected_rho"] for row in fold_summary]
        or receipt.get("oof_closure", {}).get("unique_original_iids") != 644
        or receipt.get("oof_closure", {}).get("embedded_per_iid_evidence_count") != 644
        or receipt.get("oof_closure", {}).get("each_original_evaluated_exactly_once") is not True
        or type(metrics) is not dict
        or receipt.get("qualification_scope", {}).get(
            "exposed_five_view_codec_development_gate")
            != metrics.get("exposed_five_view_codec_development_gate")
        or aggregate_result.get("receipt_sha256") != receipt_sha
        or aggregate_result.get("receipt_digest") != receipt_digest
        or aggregate_result.get("exposed_five_view_codec_development_gate")
            != metrics.get("exposed_five_view_codec_development_gate")
        or aggregate_result.get("inference_authorized") is not False
    ):
        raise SystemExit("aggregate receipt/log semantic closure differs")
    aggregate_binding = {
        "path": str((root / "aggregate/receipt.json").resolve(strict=True)),
        "sha256": receipt_sha, "receipt_digest": receipt_digest,
        "size_bytes": info.st_size,
        "exposed_five_view_codec_development_gate":
            metrics["exposed_five_view_codec_development_gate"],
    }
else:
    if passing_count == 5:
        raise SystemExit("INNER_NO_GO branch has no no-go fold")
    if (root / "aggregate").exists() or (root / "aggregate").is_symlink():
        raise SystemExit("INNER_NO_GO branch created aggregate directory")
    if any((root / f"logs/aggregate{suffix}").exists()
           for suffix in (".stdout", ".stderr")):
        raise SystemExit("INNER_NO_GO branch created aggregate logs")

actual_files = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*") if path.is_file() and not path.is_symlink()
}
actual_directories = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*") if path.is_dir() and not path.is_symlink()
}
expected_directories = {"logs", *(f"fold{index}" for index in range(5))}
if mode == "success":
    expected_directories.add("aggregate")
if actual_files != preseal_expected:
    raise SystemExit("branch preseal exact-file set differs")
if actual_directories != expected_directories:
    raise SystemExit("branch preseal exact-directory set differs")
expected_preseal_count = 29 if mode == "success" else 21 + passing_count
if len(actual_files) != expected_preseal_count:
    raise SystemExit("branch preseal exact-file count differs")
if any(
    path.is_symlink() or (not path.is_file() and not path.is_dir())
    for path in root.rglob("*")
):
    raise SystemExit("run tree contains symlink or special member")
preseal_manifest = []
for relative in sorted(actual_files):
    _, info, digest = read_sealed(
        root / relative, "preseal " + relative, allow_empty=True
    )
    preseal_manifest.append({
        "path": relative, "sha256": digest, "size_bytes": info.st_size,
        "mode_octal": "0444", "nlink": 1,
        "physical_identity": [info.st_dev, info.st_ino],
    })

qualification = {
    "exposed_five_view_codec_development_gate": (
        aggregate_binding["exposed_five_view_codec_development_gate"]
        if aggregate_binding is not None else False
    ),
    "global_exact5_not_evaluated": mode == "inner-no-go",
    "unseen_hostile_transform_gate": False,
    "unseen_hostile_transform_gate_evaluated": False,
    "scientific_confirmation_claimed": False,
    "latent_metric_qualified": False,
    "action_representation_qualified": False,
    "identity_disentanglement_qualified": False,
    "identity_preservation_qualified": False,
    "vae_necessary": None,
    "prior_qualified": False,
    "prior_generation_qualified": False,
    "generation_qualified": False,
    "renderer_qualified": False,
    "video_editing_qualified": False,
    "inference_authorized": False,
    "web_evaluation_authorized": False,
    "full644_refit_authorized": False,
}
seal = {
    "schema_version": (
        "v4f-vjepa2-residual-homotopy-exact5-parallel-run-seal-v1"
        if mode == "success"
        else "v4f-vjepa2-residual-homotopy-inner-no-go-run-seal-v1"
    ),
    "status": (
        "V4F_EXACT5_PARALLEL_SEALED"
        if mode == "success" else "V4F_EXACT5_INNER_NO_GO_SEALED"
    ),
    "branch": mode,
    "source_authority": {
        "release_tree_sha256": release_tree_sha,
        "release_manifest_sha256": release_manifest_sha,
        "release_manifest_digest": release_manifest_digest,
        "runtime_sha256": runtime_sha, "runtime_test_sha256": tests_sha,
        "python_sha256": python_sha, "controller_sha256": controller_sha,
        "checkpoint_schema": checkpoint_schema,
        "authority_snapshot": authority_snapshot,
        "bindings": source_bindings,
    },
    "cpu_postflight_gate": cpu_gate,
    "launch_plan_sha256": launch_sha,
    "folds": fold_summary,
    "passing_fold_count": passing_count,
    "inner_no_go_fold_count": 5 - passing_count,
    "per_fold_oof_evidence_counts": [
        row["oof_evidence_count"] for row in fold_summary
    ],
    "partial_pass_fold_oof_may_have_been_read": mode == "inner-no-go",
    "global_exact5_not_evaluated": mode == "inner-no-go",
    "aggregate_forbidden_and_absent": mode == "inner-no-go",
    "aggregate_receipt": aggregate_binding,
    "preseal_manifest": preseal_manifest,
    "preseal_manifest_sha256": object_sha(preseal_manifest),
    "preseal_exact_file_count": expected_preseal_count,
    "postseal_exact_file_count": expected_preseal_count + 1,
    "no_extra_files_or_symlinks": True,
    "all_directories_postseal_mode_octal": "0555",
    "all_files_postseal_mode_octal": "0444",
    "all_files_postseal_nlink": 1,
    "html_or_video_generated": False,
    "html_or_video_authorized": False,
    "qualification_scope": qualification,
}
seal["seal_digest"] = object_sha(seal)
raw = canonical(seal) + b"\n"
fd = os.open(
    seal_path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
    0o400,
)
with os.fdopen(fd, "w+b") as handle:
    os.fchmod(handle.fileno(), 0o444)
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    written = os.fstat(handle.fileno())
    handle.seek(0); readback = handle.read(); closed = os.fstat(handle.fileno())
after = seal_path.lstat()
if (
    readback != raw or seal_path.is_symlink()
    or identity(written) != identity(closed) or identity(closed) != identity(after)
    or stat.S_IMODE(after.st_mode) != 0o444 or after.st_nlink != 1
):
    raise SystemExit("branch seal create/readback differs")
for directory in sorted(
    (path for path in root.rglob("*") if path.is_dir()), reverse=True
):
    os.chmod(directory, 0o555)
os.chmod(root, 0o555)
final_files = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*") if path.is_file() and not path.is_symlink()
}
if final_files != preseal_expected | {seal_name}:
    raise SystemExit("postseal exact-file set differs")
if any(
    path.is_symlink()
    or (not path.is_dir() and not path.is_file())
    or (path.is_dir() and stat.S_IMODE(path.lstat().st_mode) != 0o555)
    or (path.is_file() and (
        stat.S_IMODE(path.lstat().st_mode) != 0o444
        or path.lstat().st_nlink != 1
    ))
    for path in [root, *root.rglob("*")]
):
    raise SystemExit("postseal tree envelope differs")
final_bindings = {}
for relative in sorted(final_files):
    raw_item, info, digest = read_sealed(
        root / relative, "postseal " + relative, allow_empty=True
    )
    final_bindings[relative] = {
        "sha256": digest, "size_bytes": info.st_size,
        "physical_identity": [info.st_dev, info.st_ino],
    }
expected_final = {
    row["path"]: {
        "sha256": row["sha256"], "size_bytes": row["size_bytes"],
        "physical_identity": row["physical_identity"],
    }
    for row in preseal_manifest
}
if (
    len(expected_final) != len(preseal_manifest)
    or set(expected_final) != preseal_expected
    or any(final_bindings.get(relative) != expected
           for relative, expected in expected_final.items())
):
    raise SystemExit("postseal files do not exactly join preseal manifest")
seal_raw_final, seal_info_final, seal_sha_final = read_sealed(
    seal_path, "final seal"
)
expected_seal_binding = {
    "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw),
    "physical_identity": [after.st_dev, after.st_ino],
}
if (
    seal_raw_final != raw
    or seal_sha_final != expected_seal_binding["sha256"]
    or seal_info_final.st_size != len(raw)
    or stat.S_IMODE(seal_info_final.st_mode) != 0o444
    or seal_info_final.st_nlink != 1
    or identity(seal_info_final) != identity(after)
    or final_bindings.get(seal_name) != expected_seal_binding
):
    raise SystemExit("postseal raw seal binding differs")
reloaded = json.loads(
    seal_raw_final,
    object_pairs_hook=pairs, parse_constant=nonfinite,
)
unsigned = dict(reloaded); reloaded_digest = unsigned.pop("seal_digest", None)
if reloaded_digest != object_sha(unsigned) or reloaded != seal:
    raise SystemExit("final seal selfdigest/readback differs")
print(json.dumps({
    "branch": mode, "seal_path": str(seal_path.resolve(strict=True)),
    "seal_sha256": final_bindings[seal_name]["sha256"],
    "seal_digest": seal["seal_digest"],
    "preseal_exact_file_count": expected_preseal_count,
    "postseal_exact_file_count": expected_preseal_count + 1,
    "passing_fold_count": passing_count,
    "inner_no_go_fold_count": 5 - passing_count,
    "final_tree_file_bindings_sha256": object_sha(final_bindings),
}, sort_keys=True, separators=(",", ":")))
PY
  )"
  [[ "${fn_result}" == \{*\} \
      && "$(printf '%s\n' "${fn_result}" | wc -l | tr -d ' ')" == 1 ]] || \
    fail "branch postflight seal result differs"
  if ! fn_post_snapshot="$(snapshot_authorities "${fn_python_bin}" "${fn_release_root}" \
       "${fn_feature_root}" "${fn_v4a_receipt}" "${fn_v4c_frontier_receipt}" \
       "${fn_v4d_receipt}" "${fn_controller_source}" "${fn_expected_controller_sha}")"; then
    fail "branch postseal read-only authority snapshot failed"
  fi
  [[ "${fn_post_snapshot}" == "${fn_pre_snapshot}" ]] || \
    fail "authority changed across branch seal"
  trap - EXIT
  cleanup_cache "${fn_cache}"
  if [[ "${fn_mode}" == success ]]; then
    printf 'V4F_FINAL_SEAL={"postflight":%s,"postflight_cache_cleaned":true,"fresh_cache_root":"%s"}\n' \
      "${fn_result}" "${fn_cache}"
  else
    printf 'V4F_INNER_NO_GO_SEAL={"postflight":%s,"postflight_cache_cleaned":true,"fresh_cache_root":"%s"}\n' \
      "${fn_result}" "${fn_cache}"
  fi
}

run_seal_child() {
  run_branch_seal_child success "$@"
}

run_no_go_seal_child() {
  run_branch_seal_child inner-no-go "$@"
}
if [[ "${1:-}" == __run_fold_child ]]; then
  shift
  run_fold_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_preflight_child ]]; then
  shift
  run_preflight_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_aggregate_child ]]; then
  shift
  run_aggregate_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_seal_child ]]; then
  shift
  run_seal_child "$@"
  exit 0
fi

if [[ "${1:-}" == __run_no_go_seal_child ]]; then
  shift
  run_no_go_seal_child "$@"
  exit 0
fi

[[ $# -eq 8 ]] || \
  fail "usage: $0 RELEASE PYTHON FEATURE_ROOT V4A_RECEIPT V4C_FRONTIER_RECEIPT V4D_RECEIPT FRESH_RUN_ROOT EXPECTED_CONTROLLER_SHA256"
readonly main_release_root=$1
readonly main_python_bin=$2
readonly main_feature_root=$3
readonly main_v4a_receipt=$4
readonly main_v4c_frontier_receipt=$5
readonly main_v4d_receipt=$6
readonly main_run_root=$7
readonly main_expected_controller_sha=$8
if ! main_controller_source="$(readlink -f -- "${BASH_SOURCE[0]}")"; then
  fail "detached controller canonical path resolution failed"
fi
[[ -n "${main_controller_source}" ]] || fail "detached controller canonical path is empty"
readonly main_controller_source
readonly main_runtime="${main_release_root}/${runtime_relative}"
readonly main_test_module=methods.bernini_action_editing.tests.test_semantic_anchor_vjepa2_multiview_global_codec_v4f_residual_homotopy

[[ "${main_expected_controller_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "controller SHA argument differs"
require_plain_file "${main_controller_source}" "${main_expected_controller_sha}" 555 detached-controller
[[ "${main_controller_source}" != "${main_release_root%/}/"* ]] || \
  fail "controller must remain detached from release"
require_plain_file "${main_python_bin}" "${expected_python_sha256}" 755 python
require_release "${main_release_root}" "${main_python_bin}"
require_sealed_json "${main_feature_root}/feature_extraction_receipt.json" \
  "${expected_feature_receipt_sha256}" feature-receipt
require_sealed_json "${main_v4a_receipt}" "${expected_v4a_receipt_sha256}" v4a-receipt
require_sealed_json "${main_v4c_frontier_receipt}" \
  "${expected_v4c_frontier_receipt_sha256}" v4c-frontier-receipt
require_sealed_json "${main_v4d_receipt}" "${expected_v4d_receipt_sha256}" v4d-receipt
[[ "${main_run_root}" == /* && "${main_run_root}" != / \
    && ! -e "${main_run_root}" && ! -L "${main_run_root}" \
    && "${main_run_root}" == "$(readlink -m -- "${main_run_root}")" ]] || \
  fail "run root is not fresh/absolute/canonical/safe"
if ! main_run_parent="$(dirname -- "${main_run_root}")"; then
  fail "run parent resolution failed"
fi
[[ -n "${main_run_parent}" ]] || fail "run parent is empty"
readonly main_run_parent
[[ -d "${main_run_parent}" && ! -L "${main_run_parent}" && -w "${main_run_parent}" \
    && "${main_run_parent}" == "$(readlink -f -- "${main_run_parent}")" ]] || \
  fail "run parent differs"

if ! main_authority_pre="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "initial authority snapshot failed"
fi
[[ "${main_authority_pre}" =~ ^[0-9a-f]{64}:[0-9a-f]{64}:[0-9a-f]{64}:[0-9a-f]{64}$ ]] || \
  fail "initial authority snapshot format differs"
readonly main_authority_pre

# This exact holder gate is also the deliberate stop point for the required
# real-entry fake-squeue smoke.  It remains before preflight srun and the first
# persistent mkdir.
for main_index in 0 1 2 3 4; do
  main_projection="$(squeue -h -j "${worker_jobs[${main_index}]}" \
    -w "${worker_nodes[${main_index}]}" -o '%T|%u' | tr -d ' ')"
  [[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
    fail "holder fold${main_index}/${worker_jobs[${main_index}]}/${worker_nodes[${main_index}]} differs"
done
main_projection="$(squeue -h -j "${preflight_job}" -w "${preflight_node}" \
  -o '%T|%u' | tr -d ' ')"
[[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
  fail "preflight holder ${preflight_job}/${preflight_node} differs"

# The expensive normal/-O suite, compilation and CLI audit run on a compute
# node.  No persistent run-root member exists until this exact CPU step passes
# and removes its node-local cache.
main_preflight_output="$(
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${preflight_job}" --nodelist="${preflight_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${preflight_cpus}" --mem="${preflight_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_preflight_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
        "${preflight_job}" "${preflight_node}" "${main_expected_controller_sha}" \
        "${main_authority_pre}"
)"
[[ "${main_preflight_output}" == V4F_PREFLIGHT_PASS=* \
    && "$(printf '%s\n' "${main_preflight_output}" | wc -l | tr -d ' ')" == 1 ]] || \
  fail "CPU exact preflight result differs"
[[ ! -e "${main_run_root}" && ! -L "${main_run_root}" ]] || \
  fail "CPU preflight created the persistent run root"
if ! main_authority_now="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "post-preflight authority snapshot failed"
fi
[[ "${main_authority_now}" == "${main_authority_pre}" ]] || \
  fail "CPU preflight changed authority"
for main_index in 0 1 2 3 4; do
  main_projection="$(squeue -h -j "${worker_jobs[${main_index}]}" \
    -w "${worker_nodes[${main_index}]}" -o '%T|%u' | tr -d ' ')"
  [[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
    fail "post-preflight holder fold${main_index} differs"
done

# First persistent mutation after every CPU/input/holder gate.
mkdir "${main_run_root}"
mkdir "${main_run_root}/logs"
chmod 0700 "${main_run_root}" "${main_run_root}/logs"
"${main_python_bin}" -I -S -B - "${main_run_root}/launch-plan.json" \
  "${main_expected_controller_sha}" "${main_authority_pre}" \
  "${main_preflight_output}" <<'PY'
from pathlib import Path
import json, os, sys
path = Path(sys.argv[1])
payload = {
    "schema_version": "v4f-exact5-parallel-launch-plan-v1",
    "controller_sha256": sys.argv[2], "authority_snapshot": sys.argv[3],
    "cpu_preflight": json.loads(sys.argv[4].split("=", 1)[1]),
    "preflight_request": {
        "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
        "ntasks": 1, "cpus": 4, "memory": "4G", "gres": "none",
        "overlap": True, "exact": True,
    },
    "source_authority": {
        "release_tree_sha256": "5282306dcb95c76e48549395d0fb030775be3990b23a043cc34a3e6d3123036d",
        "release_manifest_sha256": "7e9a0fbcbce4743a32f53335a3f268eac7c7f579614feb91b6eaa35f44b2a471",
        "release_manifest_digest": "feae69e528e28a43ddca93ae8eb06883a94d794ec1a33ce0c5ab6bc61279f19b",
        "runtime_sha256": "97cd77e64a4dfaf3036e6c50a5b85060fd616f87371e5d967e69db1170466d74",
        "runtime_test_sha256": "15b70d8f56340f553f8f6d907cc5c20e25d2b0c0ed88d0a81f2b66f5bd3ac319",
        "python_sha256": "8d7e74da2724b634c4ed5c7c5a583f21786883bc76bab93b3e381417a7f7483a",
    },
    "receipt_contract": {
        "fold_schema": "semantic-anchor-vjepa2-residual-homotopy-fold-receipt-v4f",
        "aggregate_schema": "semantic-anchor-vjepa2-residual-homotopy-exact5-receipt-v4f",
        "pass_status": "V4F_RESIDUAL_HOMOTOPY_KNOWN_EXPOSED_DEVELOPMENT",
        "inner_no_go_status": "V4F_INNER_NO_GO_OOF_UNREAD",
        "checkpoint_schema": "semantic-anchor-vjepa2-residual-homotopy-fold-checkpoint-v4f",
        "pass_fold_exact_files": ["fold.json", "preselection.pt", "selected.pt"],
        "inner_no_go_fold_exact_files": ["fold.json", "preselection.pt"],
        "partial_pass_fold_oof_may_have_been_read": True,
        "any_inner_no_go_forbids_aggregate_and_final_scientific_receipt": True,
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
    "aggregate_request": {
        "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
        "ntasks": 1, "cpus": 8, "memory": "12G", "gres": "none",
        "overlap": True, "exact": True,
    },
    "postflight_request": {
        "job": 143811, "node": "auh7-1b-gpu-306", "nodes": 1,
        "ntasks": 1, "cpus": 8, "memory": "12G", "gres": "none",
        "overlap": True, "exact": True,
    },
}
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"
descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o400)
with os.fdopen(descriptor, "w+b") as handle:
    os.fchmod(handle.fileno(), 0o444)
    handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    written = os.fstat(handle.fileno())
    handle.seek(0); readback = handle.read(); closed = os.fstat(handle.fileno())
after = path.lstat()
identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode,
                          value.st_nlink, value.st_mtime_ns, value.st_ctime_ns)
if (readback != raw or identity(written) != identity(closed) or identity(closed) != identity(after)
        or after.st_nlink != 1 or (after.st_mode & 0o7777) != 0o444 or path.is_symlink()):
    raise SystemExit("launch plan create-only same-FD seal differs")
PY

main_pids=()
main_rcs=()
for main_index in 0 1 2 3 4; do
  main_stdout="${main_run_root}/logs/fold${main_index}.stdout"
  main_stderr="${main_run_root}/logs/fold${main_index}.stderr"
  (
    set -o noclobber
    exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      srun --jobid="${worker_jobs[${main_index}]}" --nodelist="${worker_nodes[${main_index}]}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${worker_cpus}" --mem="${worker_memory}" \
        --gres=gpu:mi210:1 --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${main_controller_source}" __run_fold_child \
          "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
          "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
          "${main_run_root}/fold${main_index}" "${main_index}" \
          "${worker_jobs[${main_index}]}" "${worker_nodes[${main_index}]}" \
          "${main_expected_controller_sha}" "${main_authority_pre}" \
        >"${main_stdout}" 2>"${main_stderr}"
  ) &
  main_pids[${main_index}]=$!
done

main_any_failure=0
for main_index in 0 1 2 3 4; do
  if wait "${main_pids[${main_index}]}"; then
    main_rcs[${main_index}]=0
  else
    main_rcs[${main_index}]=$?
    main_any_failure=1
  fi
  chmod 0444 "${main_run_root}/logs/fold${main_index}.stdout" \
    "${main_run_root}/logs/fold${main_index}.stderr"
done
(( main_any_failure == 0 )) || \
  fail "one or more v4-F folds failed before a receipt branch closed; rc=${main_rcs[*]}"

main_pass_count=0
main_no_go_count=0
main_fold_branches=()
for main_index in 0 1 2 3 4; do
  main_verifier_line="$(sed -n '3p' "${main_run_root}/logs/fold${main_index}.stdout")"
  main_branch="$("${main_python_bin}" -I -S -B - "${main_verifier_line}" "${main_index}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
if value.get("fold") != int(sys.argv[2]) or value.get("branch") not in {"PASS", "INNER_NO_GO"}:
    raise SystemExit("fold verifier branch line differs")
print(value["branch"])
PY
  )"
  main_fold_branches[${main_index}]=${main_branch}
  if [[ "${main_branch}" == PASS ]]; then
    main_expected_fold_listing=$'fold.json\npreselection.pt\nselected.pt'
    ((main_pass_count += 1))
  else
    main_expected_fold_listing=$'fold.json\npreselection.pt'
    ((main_no_go_count += 1))
  fi
  if ! main_fold_listing="$(find "${main_run_root}/fold${main_index}" \
       -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)"; then
    fail "fold${main_index} exact-file scan failed"
  fi
  if ! main_fold_special="$(find "${main_run_root}/fold${main_index}" \
       -mindepth 1 -maxdepth 1 ! -type f -print -quit)"; then
    fail "fold${main_index} special-member scan failed"
  fi
  [[ -d "${main_run_root}/fold${main_index}" \
      && ! -L "${main_run_root}/fold${main_index}" \
      && "$(stat -c %a "${main_run_root}/fold${main_index}")" == 555 \
      && "${main_fold_listing}" == "${main_expected_fold_listing}" \
      && -z "${main_fold_special}" ]] || \
    fail "fold${main_index} did not return its sealed branch-specific exact closure"
done
(( main_pass_count + main_no_go_count == 5 )) || fail "fold branch census differs"
if ! main_authority_now="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "post-fold authority snapshot failed"
fi
[[ "${main_authority_now}" == "${main_authority_pre}" ]] || \
  fail "fold workers changed authority"

main_projection="$(squeue -h -j "${aggregate_job}" -w "${aggregate_node}" \
  -o '%T|%u' | tr -d ' ')"
[[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
  fail "branch postflight holder ${aggregate_job}/${aggregate_node} differs"

if (( main_no_go_count > 0 )); then
  # Per-fold procedures are intentionally independent.  Passing folds may
  # already have read their own OOF after sealing selection, but any no-go fold
  # makes global aggregate/scientific receipt creation forbidden.
  [[ ! -e "${main_run_root}/aggregate" && ! -L "${main_run_root}/aggregate" \
      && ! -e "${main_run_root}/logs/aggregate.stdout" \
      && ! -e "${main_run_root}/logs/aggregate.stderr" ]] || \
    fail "INNER_NO_GO branch has an aggregate artifact"
  main_seal_output="$(
    env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
      srun --jobid="${aggregate_job}" --nodelist="${aggregate_node}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${aggregate_cpus}" --mem="${aggregate_memory}" \
        --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
        "${main_controller_source}" __run_no_go_seal_child \
          "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
          "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
          "${main_run_root}" "${aggregate_job}" "${aggregate_node}" \
          "${main_expected_controller_sha}" "${main_authority_pre}"
  )"
  [[ "${main_seal_output}" == V4F_INNER_NO_GO_SEAL=* \
      && "$(printf '%s\n' "${main_seal_output}" | wc -l | tr -d ' ')" == 1 ]] || \
    fail "INNER_NO_GO compute postflight result differs"
  if ! main_bad_directories="$(find "${main_run_root}" -type d ! -perm 0555 -print -quit)"; then
    fail "INNER_NO_GO final directory scan failed"
  fi
  [[ "$(stat -c '%a:%h' "${main_run_root}/inner-no-go-seal.json")" == 444:1 \
      && ! -L "${main_run_root}/inner-no-go-seal.json" \
      && "$(stat -c %a "${main_run_root}")" == 555 \
      && -z "${main_bad_directories}" ]] || \
    fail "INNER_NO_GO final sealed run envelope differs"
  printf '%s\n' "${main_seal_output}"
  printf 'V4F_EXACT5_INNER_NO_GO_COMPLETE pass_folds=%s no_go_folds=%s run_root=%s\n' \
    "${main_pass_count}" "${main_no_go_count}" "${main_run_root}"
  exit 0
fi

# All five fold-local nested gates passed.  Only now may a separate exact
# CPU-only step load/replay the ten checkpoints and aggregate exact644 OOF.
[[ "${main_pass_count}" == 5 ]] || fail "success branch pass count differs"
mkdir "${main_run_root}/aggregate"
chmod 0700 "${main_run_root}/aggregate"
main_aggregate_stdout="${main_run_root}/logs/aggregate.stdout"
main_aggregate_stderr="${main_run_root}/logs/aggregate.stderr"
main_aggregate_rc=0
(
  set -o noclobber
  exec env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${aggregate_job}" --nodelist="${aggregate_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${aggregate_cpus}" --mem="${aggregate_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_aggregate_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
        "${main_run_root}" "${aggregate_job}" "${aggregate_node}" \
        "${main_expected_controller_sha}" "${main_authority_pre}" \
      >"${main_aggregate_stdout}" 2>"${main_aggregate_stderr}"
) || main_aggregate_rc=$?
chmod 0444 "${main_aggregate_stdout}" "${main_aggregate_stderr}"
(( main_aggregate_rc == 0 )) || \
  fail "v4-F CPU aggregate child failed with status ${main_aggregate_rc}"
if ! main_authority_now="$(snapshot_authorities "${main_python_bin}" "${main_release_root}" \
     "${main_feature_root}" "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" \
     "${main_v4d_receipt}" "${main_controller_source}" "${main_expected_controller_sha}")"; then
  fail "post-aggregate authority snapshot failed"
fi
[[ "${main_authority_now}" == "${main_authority_pre}" ]] || \
  fail "CPU aggregate changed authority"

# A second exact CPU-only step performs exact29-to-exact30 same-FD readback and
# creates the final run seal.  The controller/login process never loads a model.
main_projection="$(squeue -h -j "${aggregate_job}" -w "${aggregate_node}" \
  -o '%T|%u' | tr -d ' ')"
[[ "${main_projection}" == "RUNNING|guangyi.chen" ]] || \
  fail "success postflight holder ${aggregate_job}/${aggregate_node} differs"
main_seal_output="$(
  env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
    srun --jobid="${aggregate_job}" --nodelist="${aggregate_node}" \
      --nodes=1 --ntasks=1 --cpus-per-task="${aggregate_cpus}" --mem="${aggregate_memory}" \
      --gres=none --overlap --exact --kill-on-bad-exit=1 --immediate=60 \
      "${main_controller_source}" __run_seal_child \
        "${main_release_root}" "${main_python_bin}" "${main_feature_root}" \
        "${main_v4a_receipt}" "${main_v4c_frontier_receipt}" "${main_v4d_receipt}" \
        "${main_run_root}" "${aggregate_job}" "${aggregate_node}" \
        "${main_expected_controller_sha}" "${main_authority_pre}"
)"
[[ "${main_seal_output}" == V4F_FINAL_SEAL=* \
    && "$(printf '%s\n' "${main_seal_output}" | wc -l | tr -d ' ')" == 1 ]] || \
  fail "final compute postflight result differs"
if ! main_bad_directories="$(find "${main_run_root}" -type d ! -perm 0555 -print -quit)"; then
  fail "success final directory scan failed"
fi
[[ "$(stat -c '%a:%h' "${main_run_root}/seal.json")" == 444:1 \
    && ! -L "${main_run_root}/seal.json" \
    && "$(stat -c %a "${main_run_root}")" == 555 \
    && -z "${main_bad_directories}" ]] || \
  fail "final sealed run envelope differs"
printf '%s\n' "${main_seal_output}"
printf 'V4F_EXACT5_PARALLEL_COMPLETE run_root=%s\n' "${main_run_root}"

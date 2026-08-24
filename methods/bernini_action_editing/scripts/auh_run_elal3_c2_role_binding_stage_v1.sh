#!/usr/bin/env bash
# Run exactly one node-local C2 stage.  Assets must have been transported over
# this holder's srun stdin and rehashed inside the compute step.  This launcher
# never reads a login-node release path and never reuses the materializer's
# mode-0644 runtime tree.

set -Eeuo pipefail
umask 0027

fail() { echo "[elal3-c2-role-stage] ERROR: $*" >&2; exit 2; }

# Frozen final trainer/tool/exact16 release literals.  Malformed or altered
# values block before model import or output creation.
expected_archive_sha256="e6ccc7c55c50d03d6df57cb8a9a3d85bb2dc1b0977ef1905105944757b720e61"
expected_archive_size="1054720"
expected_manifest_sha256="4e95f179a6274bca5611a0532402a71d23db15822e14b85d5111309f59246f15"
expected_manifest_size="8830"
expected_runner_sha256="63f35b39e60dbf2c1dd1dcecb29393c04d9f00fd0833054e7d81d40790dfe4ce"
expected_runner_size="447559"
expected_origin_verifier_sha256="07122fd71e8f170b5a50761255a664ac17fc2c66b7b8970a1c113bc8d5e605c1"
expected_origin_verifier_size="24717"
expected_gate_controller_sha256="f4e931b1f50473a9391aa7e7e68464213aaf43e85cc5a8bee792c380c2035af1"
expected_gate_controller_size="28107"
expected_bundle_sha256="b31d5e1594a112f965a3cebd527d5189a561e2cc2d83cfe94014872ffb94d1b8"
expected_bundle_size="78277976"
expected_receipt_sha256="a1ca0d3c015a54d61c8a71d00bc78688dab20d6592ba30ddf73b0ea18e7d70ee"
expected_receipt_size="52752"
expected_receipt_digest="225255f5ada73848686b240c4a53001c9dd65b1373da2b293c2da8c2ec14f35d"

sha_re='^[0-9a-f]{64}$'
uint_re='^[1-9][0-9]*$'
for value in \
  "${expected_archive_sha256}" "${expected_manifest_sha256}" \
  "${expected_runner_sha256}" "${expected_origin_verifier_sha256}" \
  "${expected_gate_controller_sha256}" "${expected_bundle_sha256}" \
  "${expected_receipt_sha256}" "${expected_receipt_digest}"; do
  [[ "${value}" =~ ${sha_re} ]] || fail "release literal is still PENDING"
done
for value in \
  "${expected_archive_size}" "${expected_manifest_size}" \
  "${expected_runner_size}" "${expected_origin_verifier_size}" \
  "${expected_gate_controller_size}" "${expected_bundle_size}" \
  "${expected_receipt_size}"; do
  [[ "${value}" =~ ${uint_re} ]] || fail "release size literal is still PENDING"
done

[[ "${ELAL3_C2_STAGE:-}" =~ ^(preflight|fresh1|exact10)$ ]] || fail "ELAL3_C2_STAGE must be preflight, fresh1, or exact10"
[[ "${ELAL3_C2_LAUNCHER_SHA256:-}" =~ ${sha_re} ]] || fail "outer controller did not pin launcher SHA"
[[ "${ELAL3_C2_LAUNCHER_SIZE:-}" =~ ${uint_re} ]] || fail "outer controller did not pin launcher size"
stage="${ELAL3_C2_STAGE}"

# /tmp is node-local.  The identical release-scoped string on all three
# holders keeps sealed control paths replayable while never creating three
# concurrent roots in a shared /vast namespace.
node_root="/tmp/elal3-c2-role-e6ccc7c5-v10"
assets="${node_root}/assets"
archive="${assets}/source.tar"
manifest="${assets}/source.manifest.json"
bundle="${assets}/c2-exact16-latents.safetensors"
launcher="${assets}/auh_run_elal3_c2_role_binding_stage_v1.sh"
launcher_self="$(readlink -f "$0")"
[[ "${launcher_self}" == "${launcher}" ]] || fail "launcher actual path differs from staged canonical path"
[[ "$(stat -c '%s:%a:%h' "${launcher_self}")" == "${ELAL3_C2_LAUNCHER_SIZE}:444:1" ]] || fail "launcher self size/mode/link differs"
[[ "$(sha256sum "${launcher_self}" | awk '{print $1}')" == "${ELAL3_C2_LAUNCHER_SHA256}" ]] || fail "launcher self SHA differs"

python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
bernini_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
veomni_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
checkpoint_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
packet_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/elal3_c1_oracle_diagnostic_preflight_20260817_v1/simulator_gt_canary_v1"
packet_manifest_sha256="2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"

job_id="${SLURM_JOB_ID:-}"
node="${HOSTNAME%%.*}"
case "${job_id}:${node}" in
  141620:auh7-1b-gpu-226)
    arm_id="A_duplicate_control"; seed=20260821; master_port=29720 ;;
  141618:auh7-1b-gpu-249)
    arm_id="B_paired_role"; seed=20260821; master_port=29718 ;;
  141619:auh7-1b-gpu-257)
    arm_id="B_paired_role_replica"; seed=20260822; master_port=29719 ;;
  *) fail "unregistered holder/node placement: ${job_id:-unset}:${node}" ;;
esac

for path in "${archive}" "${manifest}" "${bundle}" "${launcher}" "${python_bin}" "${packet_root}/manifest.json"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "node-local required file unavailable: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint_root}" "${packet_root}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "node-local required directory unavailable: ${path}"
done
[[ -x "${python_bin}" ]] || fail "pinned Python is not executable"
[[ "$(stat -c '%s:%a:%h' "${archive}")" == "${expected_archive_size}:444:1" ]] || fail "node-local source archive size/mode/link differs"
[[ "$(stat -c '%s:%a:%h' "${manifest}")" == "${expected_manifest_size}:444:1" ]] || fail "node-local source manifest size/mode/link differs"
[[ "$(stat -c '%s:%a:%h' "${bundle}")" == "${expected_bundle_size}:444:1" ]] || fail "node-local exact16 bundle size/mode/link differs"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${expected_archive_sha256}" ]] || fail "node-local source archive SHA differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${expected_manifest_sha256}" ]] || fail "node-local source manifest SHA differs"
[[ "$(sha256sum "${bundle}" | awk '{print $1}')" == "${expected_bundle_sha256}" ]] || fail "node-local exact16 bundle SHA differs"
[[ "$(sha256sum "${packet_root}/manifest.json" | awk '{print $1}')" == "${packet_manifest_sha256}" ]] || fail "node-local simulator packet SHA differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch="$(mktemp -d "${scratch_parent%/}/elal3-c2-role-${stage}.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  case "${scratch}" in "${scratch_parent%/}/elal3-c2-role-${stage}."*) ;; *) exit 2 ;; esac
  if [[ -d "${scratch}" && ! -L "${scratch}" ]]; then
    chmod -R u+w -- "${scratch}" || status=2
    rm -rf -- "${scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Make two independent extracts.  Both first replay the exact20 archive as an
# all-0444 source closure.  source-archive remains pristine; only the exact
# three registered authority/contract files in source are projected below.
for extract_name in source-archive source; do
mkdir -- "${scratch}/${extract_name}"
"${python_bin}" -I -B - \
  "${archive}" "${manifest}" "${scratch}/${extract_name}" \
  "${expected_archive_sha256}" "${expected_archive_size}" \
  "${expected_manifest_sha256}" "${expected_manifest_size}" <<'PY'
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive, manifest, output = map(Path, sys.argv[1:4])
archive_sha, archive_size, manifest_sha, manifest_size = sys.argv[4:8]
def reject(message): raise SystemExit("training release rejected: " + message)
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
def sha(raw): return hashlib.sha256(raw).hexdigest()
manifest_raw = manifest.read_bytes()
if len(manifest_raw) != int(manifest_size) or sha(manifest_raw) != manifest_sha: reject("manifest outer binding differs")
try: value = json.loads(manifest_raw)
except Exception as error: reject("manifest JSON parse failed: " + str(error))
if manifest_raw != canonical(value) + b"\n": reject("manifest is not canonical JSON+newline")
unsigned = dict(value); digest = unsigned.pop("manifest_digest", None)
if digest != sha(canonical(unsigned)): reject("manifest self digest differs")
if value.get("schema_version") != "bernini-elal3-c2-role-binding-training-release-v1": reject("manifest schema differs")
if value.get("archive_format") != "fixed-ustar-ascii-sorted-owner0-mtime0-record10240-v1": reject("archive format differs")
if value.get("archive_member_mode") != "0444" or value.get("fresh_training_runtime_file_mode") != "0444" or value.get("fresh_training_runtime_root_mode") != "0555" or value.get("materializer_runtime_tree_reuse_forbidden") is not True: reject("initial source/archive extract mode/isolation contract differs")
if value.get("stage_sequence") != ["exact3_preflight_no_update", "cross_arm_preflight_gate", "exact3_fresh1", "fresh1_acceptance_gate", "exact3_fresh_exact10"] or value.get("gate_failure_stops_later_stages") is not True or value.get("exact10_resume_from_fresh1_forbidden") is not True: reject("staged gate sequence differs")
if any(value.get(key) is not False for key in ("formal_c2_authorized", "exact160_authorized", "source_instruction_inference_authorized", "real_video_generalization_authorized", "scientific_claim_authorized")): reject("forbidden claim authorized")
archive_raw = archive.read_bytes()
if len(archive_raw) != int(archive_size) or sha(archive_raw) != archive_sha or archive_sha != value.get("archive_sha256") or len(archive_raw) != value.get("archive_size"): reject("archive binding differs")
rows = value.get("files")
if not isinstance(rows, list) or len(rows) != value.get("file_count"): reject("file closure differs")
expected = {row.get("path"): row for row in rows if isinstance(row, dict)}
if len(expected) != len(rows): reject("duplicate file rows")
with tarfile.open(archive, "r:") as source:
    members = source.getmembers(); names = [member.name for member in members]
    if names != sorted(names, key=lambda item: item.encode("ascii")) or set(names) != set(expected): reject("archive member order/closure differs")
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.isreg() or member.mode != 0o444 or member.uid != 0 or member.gid != 0 or member.mtime != 0: reject("unsafe archive member: " + member.name)
        stream = source.extractfile(member)
        if stream is None: reject("member payload absent")
        raw = stream.read(); row = expected[member.name]
        if set(row) != {"path", "sha256", "size", "mode"} or row["mode"] != "0444" or len(raw) != row["size"] or sha(raw) != row["sha256"]: reject("member binding differs: " + member.name)
        target = output.joinpath(*pure.parts); target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
        try:
            view = memoryview(raw)
            while view:
                count = os.write(descriptor, view)
                if count <= 0: reject("member write stalled")
                view = view[count:]
            os.fchmod(descriptor, 0o444); os.fsync(descriptor)
        finally: os.close(descriptor)
for root, directories, files in os.walk(output, topdown=False):
    for name in directories: os.chmod(Path(root) / name, 0o555)
os.chmod(output, 0o555)
for path in output.rglob("*"):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode): reject("symlink after extraction")
    expected_mode = 0o555 if stat.S_ISDIR(info.st_mode) else 0o444
    if stat.S_IMODE(info.st_mode) != expected_mode or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1): reject("extracted source type/mode/link differs")
PY
done

# BEGIN_ELAL3_C2_EXACT3_CONTROL_PROJECTION_V1
"${python_bin}" -I -B - \
  "${scratch}/source-archive" "${scratch}/source" \
  "md/action_editing/20260817_box/evidence/elal3_c2_simulator_optimizer_diagnostic_authority_v1.json" \
  "md/action_editing/20260817_box/evidence/elal3_c2_real_model_authority_v1.json" \
  "md/action_editing/20260817_box/evidence/elal3_c2_role_binding_experiment_contract_v1.json" <<'PY_PROJECTION'
import hashlib, os, stat, sys
from pathlib import Path, PurePosixPath

pristine_root, consumer_root = map(Path, sys.argv[1:3])
control_relpaths = tuple(sys.argv[3:])
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY = getattr(os, "O_DIRECTORY", 0)

def reject(message):
    raise SystemExit("training control projection rejected: " + message)

if len(control_relpaths) != 3 or len(set(control_relpaths)) != 3:
    reject("exact-three control path closure differs")

def tree_rows(root):
    root_info = root.lstat()
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_IMODE(root_info.st_mode) != 0o555:
        reject("extract root type/mode differs")
    rows = {}
    for current, directories, files in os.walk(root):
        current_path = Path(current)
        current_info = current_path.lstat()
        if not stat.S_ISDIR(current_info.st_mode) or stat.S_IMODE(current_info.st_mode) != 0o555:
            reject("extract directory type/mode differs")
        for name in directories:
            info = (current_path / name).lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o555:
                reject("extract child directory type/mode differs")
        for name in files:
            path = current_path / name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                reject("extract regular-file/link closure differs")
            rel = path.relative_to(root).as_posix()
            if rel in rows:
                reject("duplicate extract relative path")
            rows[rel] = info
    return rows

def held_file(root, relpath):
    pure = PurePosixPath(relpath)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        reject("unsafe control relative path")
    held = []
    try:
        root_fd = os.open(root, os.O_RDONLY | DIRECTORY | NOFOLLOW)
        held.append(root_fd)
        current = root_fd
        for part in pure.parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | DIRECTORY | NOFOLLOW, dir_fd=current)
            held.append(next_fd)
            current = next_fd
        named = os.stat(pure.name, dir_fd=current, follow_symlinks=False)
        descriptor = os.open(pure.name, os.O_RDONLY | NOFOLLOW, dir_fd=current)
        held.append(descriptor)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            reject("control file type/link differs")
        if (named.st_dev, named.st_ino, named.st_size, named.st_nlink) != (
            opened.st_dev, opened.st_ino, opened.st_size, opened.st_nlink
        ):
            reject("control named/open identity differs")
        return held, descriptor, opened
    except Exception:
        for fd in reversed(held):
            os.close(fd)
        raise

def read_twice(descriptor):
    outputs = []
    for _ in range(2):
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        outputs.append(b"".join(chunks))
    if outputs[0] != outputs[1]:
        reject("held-FD replay differs")
    return outputs[0]

initial_pristine = tree_rows(pristine_root)
initial_consumer = tree_rows(consumer_root)
if set(initial_pristine) != set(initial_consumer):
    reject("independent extract member closure differs")
if not set(control_relpaths).issubset(initial_pristine):
    reject("registered exact-three control is absent")
for relpath, pristine_info in initial_pristine.items():
    consumer_info = initial_consumer[relpath]
    if stat.S_IMODE(pristine_info.st_mode) != 0o444 or stat.S_IMODE(consumer_info.st_mode) != 0o444:
        reject("pre-projection source/archive file mode differs")
    if (pristine_info.st_dev, pristine_info.st_ino) == (consumer_info.st_dev, consumer_info.st_ino):
        reject("source/archive and consumer extracts are not independent")

for relpath in control_relpaths:
    pristine_held, pristine_fd, pristine_before = held_file(pristine_root, relpath)
    consumer_held, consumer_fd, consumer_before = held_file(consumer_root, relpath)
    try:
        if stat.S_IMODE(pristine_before.st_mode) != 0o444 or stat.S_IMODE(consumer_before.st_mode) != 0o444:
            reject("control pre-projection mode differs")
        pristine_raw = read_twice(pristine_fd)
        consumer_raw = read_twice(consumer_fd)
        if pristine_raw != consumer_raw:
            reject("control pre-projection payload differs")
        expected_size = len(pristine_raw)
        expected_sha = hashlib.sha256(pristine_raw).hexdigest()
        if pristine_before.st_size != expected_size or consumer_before.st_size != expected_size:
            reject("control pre-projection size differs")
        if hashlib.sha256(consumer_raw).hexdigest() != expected_sha:
            reject("control pre-projection SHA differs")
        os.fchmod(consumer_fd, 0o644)
        os.fsync(consumer_fd)
        consumer_after = os.fstat(consumer_fd)
        before_identity = (
            consumer_before.st_dev, consumer_before.st_ino, consumer_before.st_size,
            consumer_before.st_nlink,
        )
        after_identity = (
            consumer_after.st_dev, consumer_after.st_ino, consumer_after.st_size,
            consumer_after.st_nlink,
        )
        if before_identity != after_identity or stat.S_IMODE(consumer_after.st_mode) != 0o644:
            reject("control post-projection identity/mode differs")
        post_raw = read_twice(consumer_fd)
        if len(post_raw) != expected_size or hashlib.sha256(post_raw).hexdigest() != expected_sha or post_raw != consumer_raw:
            reject("control post-projection SHA/size replay differs")
        named_after = os.stat(
            PurePosixPath(relpath).name,
            dir_fd=consumer_held[-2],
            follow_symlinks=False,
        )
        if (
            named_after.st_dev, named_after.st_ino, named_after.st_size,
            named_after.st_nlink, stat.S_IMODE(named_after.st_mode),
        ) != (*after_identity, 0o644):
            reject("control post-projection named identity differs")
    finally:
        for fd in reversed(consumer_held):
            os.close(fd)
        for fd in reversed(pristine_held):
            os.close(fd)

final_pristine = tree_rows(pristine_root)
final_consumer = tree_rows(consumer_root)
if set(final_pristine) != set(initial_pristine) or set(final_consumer) != set(initial_consumer):
    reject("post-projection member closure differs")
for relpath in final_pristine:
    pristine_info = final_pristine[relpath]
    consumer_info = final_consumer[relpath]
    if stat.S_IMODE(pristine_info.st_mode) != 0o444:
        reject("pristine source/archive file became mutable")
    expected_consumer_mode = 0o644 if relpath in control_relpaths else 0o444
    if stat.S_IMODE(consumer_info.st_mode) != expected_consumer_mode:
        reject("consumer exact-three-only mode closure differs")
    pristine_raw = (pristine_root / relpath).read_bytes()
    consumer_raw = (consumer_root / relpath).read_bytes()
    if pristine_raw != consumer_raw:
        reject("post-projection source/consumer payload differs")
PY_PROJECTION
# END_ELAL3_C2_EXACT3_CONTROL_PROJECTION_V1

method_root="${scratch}/source/methods/bernini_action_editing"
trainer="${method_root}/train_elal3_c2_simulator_role_pair_v1.py"
derivative="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"
model_authority="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_real_model_authority_v1.json"
experiment_contract="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_role_binding_experiment_contract_v1.json"
latent_receipt="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_exact16_latent_bundle_receipt_v1.json"
materializer_run_complete="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_exact16_materializer_run_complete_v1.json"
checkpoint_exact23_manifest="${scratch}/source/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
materialize_vae_source="${method_root}/tools/materialize_vae.py"
renderer_dataset_source="${method_root}/tools/build_renderer_dataset.py"
tools_package_marker="${method_root}/tools/__init__.py"
for mutable_control in "${derivative}" "${model_authority}" "${experiment_contract}"; do
  [[ -f "${mutable_control}" && ! -L "${mutable_control}" && "$(stat -c '%a:%h' "${mutable_control}")" == "644:1" ]] || fail "projected exact-three control mode/link differs"
done
[[ "$(stat -c '%s:%a:%h' "${trainer}")" == "${expected_runner_size}:444:1" ]] || fail "fresh trainer source size/mode/link differs"
[[ "$(sha256sum "${trainer}" | awk '{print $1}')" == "${expected_runner_sha256}" ]] || fail "fresh trainer source SHA differs"
[[ "$(stat -c '%s:%a:%h' "${materialize_vae_source}")" == "32195:444:1" ]] || fail "fresh materialize_vae source size/mode/link differs"
[[ "$(sha256sum "${materialize_vae_source}" | awk '{print $1}')" == "a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0" ]] || fail "fresh materialize_vae source SHA differs"
[[ "$(stat -c '%s:%a:%h' "${renderer_dataset_source}")" == "31012:444:1" ]] || fail "fresh renderer dataset source size/mode/link differs"
[[ "$(sha256sum "${renderer_dataset_source}" | awk '{print $1}')" == "afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5" ]] || fail "fresh renderer dataset source SHA differs"
[[ "$(stat -c '%s:%a:%h' "${tools_package_marker}")" == "0:444:1" ]] || fail "fresh tools package marker size/mode/link differs"
[[ "$(sha256sum "${tools_package_marker}" | awk '{print $1}')" == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ]] || fail "fresh tools package marker SHA differs"
[[ "$(stat -c '%s:%a:%h' "${latent_receipt}")" == "${expected_receipt_size}:444:1" ]] || fail "fresh exact16 receipt size/mode/link differs"
[[ "$(sha256sum "${latent_receipt}" | awk '{print $1}')" == "${expected_receipt_sha256}" ]] || fail "fresh exact16 receipt SHA differs"

# Resolve the materializer's absolute-import dependencies from only the
# freshly extracted method root.  This rejects an archive that would fall
# back to an ambient checkout/PYTHONPATH at trainer import time.
"${python_bin}" -I -B - "${method_root}" <<'PY'
import importlib, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
materializer = importlib.import_module("materialize_elal3_simulator_c2_vae_v1")
actual = {
    "tools.materialize_vae": Path(materializer.materialize_vae.__file__).resolve(strict=True),
    "tools.build_renderer_dataset": Path(materializer.materialize_vae.raw_builder.__file__).resolve(strict=True),
}
expected = {
    "tools.materialize_vae": root / "tools/materialize_vae.py",
    "tools.build_renderer_dataset": root / "tools/build_renderer_dataset.py",
}
if actual != expected:
    raise SystemExit("isolated extracted transitive import closure differs")
PY

controls="${node_root}/controls"
own_preflight="${controls}/preflight/${arm_id}.json"
cross_gate="${controls}/gates/cross_arm_preflight_gate.json"
fresh1_gate="${controls}/gates/fresh1_acceptance_gate.json"
declare -a gate_flags=()
case "${stage}" in
  preflight)
    steps=1; stage_flags=(--preflight-only); receipt_name="PRECHECK_RECEIPT.json" ;;
  fresh1)
    steps=1; stage_flags=()
    [[ "${ELAL3_C2_OWN_PREFLIGHT_SHA256:-}" =~ ${sha_re} ]] || fail "fresh1 own-preflight SHA is required"
    [[ "${ELAL3_C2_CROSS_GATE_SHA256:-}" =~ ${sha_re} ]] || fail "fresh1 cross-gate SHA is required"
    gate_flags=(
      --own-preflight-receipt "${own_preflight}"
      --expected-own-preflight-receipt-sha256 "${ELAL3_C2_OWN_PREFLIGHT_SHA256}"
      --cross-arm-preflight-gate "${cross_gate}"
      --expected-cross-arm-preflight-gate-sha256 "${ELAL3_C2_CROSS_GATE_SHA256}"
    )
    receipt_name="TRAINING_RECEIPT.json" ;;
  exact10)
    steps=10; stage_flags=()
    [[ "${ELAL3_C2_OWN_PREFLIGHT_SHA256:-}" =~ ${sha_re} ]] || fail "exact10 own-preflight SHA is required"
    [[ "${ELAL3_C2_CROSS_GATE_SHA256:-}" =~ ${sha_re} ]] || fail "exact10 cross-gate SHA is required"
    [[ "${ELAL3_C2_FRESH1_GATE_SHA256:-}" =~ ${sha_re} ]] || fail "exact10 fresh1-gate SHA is required"
    gate_flags=(
      --own-preflight-receipt "${own_preflight}"
      --expected-own-preflight-receipt-sha256 "${ELAL3_C2_OWN_PREFLIGHT_SHA256}"
      --cross-arm-preflight-gate "${cross_gate}"
      --expected-cross-arm-preflight-gate-sha256 "${ELAL3_C2_CROSS_GATE_SHA256}"
      --fresh1-acceptance-gate "${fresh1_gate}"
      --expected-fresh1-acceptance-gate-sha256 "${ELAL3_C2_FRESH1_GATE_SHA256}"
      --fresh1-origin-verifier-name "elal3_c2_origin_receipt_verifier_v1.py"
      --expected-fresh1-origin-verifier-sha256 "${expected_origin_verifier_sha256}"
      --expected-fresh1-origin-verifier-size "${expected_origin_verifier_size}"
      --fresh1-gate-controller-name "elal3_c2_staged_gate_controller_v1.py"
      --expected-fresh1-gate-controller-sha256 "${expected_gate_controller_sha256}"
      --expected-fresh1-gate-controller-size "${expected_gate_controller_size}"
    )
    receipt_name="TRAINING_RECEIPT.json" ;;
esac
if [[ "${stage}" != preflight ]]; then
  [[ -f "${own_preflight}" && ! -L "${own_preflight}" && "$(stat -c '%a:%h' "${own_preflight}")" == "444:1" ]] || fail "sealed own preflight control is unavailable"
  [[ -f "${cross_gate}" && ! -L "${cross_gate}" && "$(stat -c '%a:%h' "${cross_gate}")" == "444:1" ]] || fail "sealed cross-arm gate is unavailable"
  [[ "$(sha256sum "${own_preflight}" | awk '{print $1}')" == "${ELAL3_C2_OWN_PREFLIGHT_SHA256}" ]] || fail "own preflight control SHA differs"
  [[ "$(sha256sum "${cross_gate}" | awk '{print $1}')" == "${ELAL3_C2_CROSS_GATE_SHA256}" ]] || fail "cross-arm gate SHA differs"
fi
if [[ "${stage}" == exact10 ]]; then
  [[ -f "${fresh1_gate}" && ! -L "${fresh1_gate}" && "$(stat -c '%a:%h' "${fresh1_gate}")" == "444:1" ]] || fail "sealed fresh1 gate is unavailable"
  [[ "$(sha256sum "${fresh1_gate}" | awk '{print $1}')" == "${ELAL3_C2_FRESH1_GATE_SHA256}" ]] || fail "fresh1 gate SHA differs"
fi

output="${node_root}/runs/${arm_id}/elal3_c2_${stage}"
log="${node_root}/runs/${arm_id}/elal3_c2_${stage}.log"
[[ ! -e "${output}" && ! -L "${output}" && ! -e "${log}" && ! -L "${log}" ]] || fail "stage output/log must be fresh"
mkdir -p -- "$(dirname -- "${output}")"

rank_wrapper="${scratch}/rank_exec.sh"
"${python_bin}" -I -B - "${rank_wrapper}" <<'PY'
import os, sys
from pathlib import Path
path = Path(sys.argv[1])
raw = b'''#!/usr/bin/env bash
set -Eeuo pipefail
umask 0077
cache_base="${1:?cache base missing}"; shift
rank="${LOCAL_RANK:?LOCAL_RANK missing}"
[[ "${rank}" =~ ^[0-7]$ ]] || exit 2
root="${cache_base}/rank_${rank}"
mkdir -m 0700 -- "${root}" "${root}/miopen-user" "${root}/miopen-custom" "${root}/torch" "${root}/xdg" "${root}/triton"
export MIOPEN_USER_DB_PATH="${root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${root}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${root}/torch"
export XDG_CACHE_HOME="${root}/xdg"
export TRITON_CACHE_DIR="${root}/triton"
exec "$@"
'''
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o500)
try:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0: raise SystemExit("rank wrapper write stalled")
        view = view[written:]
    os.fchmod(fd, 0o500); os.fsync(fd)
finally: os.close(fd)
PY
rank_cache="${scratch}/rank-cache"
mkdir -m 0700 -- "${rank_cache}"

ack_flags=(
  --ack-simulator-oracle-q-diagnostic-only
  --ack-not-source-instruction-inference
  --ack-not-formal-c2
  --ack-not-exact160
  --ack-no-real-video-or-scientific-claim
)
set +e
"${python_bin}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=8 --master-port="${master_port}" --no-python \
  "${rank_wrapper}" "${rank_cache}" "${python_bin}" -B "${trainer}" \
    --arm-id "${arm_id}" \
    --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint_root}" --packet-root "${packet_root}" \
    --latent-bundle "${bundle}" --expected-latent-bundle-sha256 "${expected_bundle_sha256}" \
    --latent-bundle-receipt "${latent_receipt}" --expected-latent-bundle-receipt-sha256 "${expected_receipt_sha256}" \
    --materializer-run-complete "${materializer_run_complete}" --expected-materializer-run-complete-sha256 "c6eee4766943c7959a2c1ad9b8b6b4e823dec054b31d2fdfb5d03aacd9f7e1ac" \
    --checkpoint-exact23-manifest "${checkpoint_exact23_manifest}" --expected-checkpoint-exact23-manifest-sha256 "a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831" \
    --external-authority "${derivative}" --expected-external-authority-sha256 "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a" \
    --model-authority "${model_authority}" --expected-model-authority-sha256 "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d" \
    --experiment-contract "${experiment_contract}" --expected-experiment-contract-sha256 "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8" \
    --output "${output}" --max-steps "${steps}" --seed "${seed}" \
    --expected-runner-source-sha256 "${expected_runner_sha256}" \
    --expected-c1-trainer-source-sha256 "521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3" \
    --expected-elal3-core-source-sha256 "70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862" \
    --expected-c2-label-source-sha256 "1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11" \
    --expected-c2-materializer-source-sha256 "b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f" \
    --expected-train-lora-source-sha256 "630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5" \
    --expected-packed-lora-source-sha256 "61c1e1076efc897d3622153d1e73eeeaf17631709f479925d3996e479cb439d6" \
    --expected-runtime-source-sha256 "62df125ac130697b03aaea167b17a02d7fcb9d766a72f0bef71037924114e59f" \
    --expected-sigma-source-sha256 "e3782a22130c09a48dc3ea27fa219af6caca445e1fce2c8f3bca7cde6058afd3" \
    "${stage_flags[@]}" "${gate_flags[@]}" "${ack_flags[@]}" >"${log}" 2>&1
status=$?
set -e
chmod 0444 -- "${log}"
[[ "${status}" -eq 0 ]] || fail "${stage} stopped with status ${status}; no later gate/stage is authorized"
receipt="${output}/${receipt_name}"
[[ -f "${receipt}" && ! -L "${receipt}" && "$(stat -c '%a:%h' "${receipt}")" == "444:1" ]] || fail "sealed ${stage} receipt missing"

"${python_bin}" -I -B - "${receipt}" "${stage}" "${arm_id}" "${job_id}" "${node}" "${seed}" "${expected_runner_sha256}" "${expected_bundle_sha256}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[1]); stage, arm, job, node, seed, runner, bundle = sys.argv[2:9]
raw = path.read_bytes(); value = json.loads(raw)
canonical = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
if raw != canonical(value) + b"\n": raise SystemExit("stage receipt is not canonical JSON+newline")
unsigned = dict(value); digest = unsigned.pop("receipt_digest", None)
if digest != hashlib.sha256(canonical(unsigned)).hexdigest(): raise SystemExit("stage receipt digest differs")
if (value.get("arm_id"), value.get("holder_job_id"), value.get("node"), value.get("seed")) != (arm, job, node, int(seed)): raise SystemExit("stage receipt placement differs")
if value.get("runner_source_sha256") != runner or value.get("latent_bundle_sha256") != bundle: raise SystemExit("stage receipt release binding differs")
if stage == "preflight":
    if value.get("schema_version") != "bernini-elal3-c2-simulator-role-pair-preflight-receipt-v1" or value.get("status") != "PRECHECK_COMPLETE_NO_OPTIMIZER_NO_UPDATE" or value.get("completed_optimizer_steps") != 0 or value.get("optimizer_constructed") is not False or value.get("all_preflight_hard_gates_pass") is not True: raise SystemExit("preflight hard gate differs")
elif stage == "fresh1":
    if value.get("status") != "FRESH1_ENGINEERING_ACCEPTANCE_COMPLETE" or value.get("completed_optimizer_steps") != 1 or value.get("all_fresh1_acceptance_gates_pass") is not True: raise SystemExit("fresh1 hard gate differs")
else:
    if value.get("status") != "EXACT10_LATENT_GATES_PASS_DECODED_REVIEW_PENDING" or value.get("completed_optimizer_steps") != 10 or value.get("latent_hard_gates_pass") is not True or value.get("resume_consumed") is not False or value.get("fresh1_checkpoint_consumed") is not False: raise SystemExit("exact10 latent hard gate differs")
print(hashlib.sha256(raw).hexdigest())
PY
echo "[elal3-c2-role-stage] COMPLETE stage=${stage} arm=${arm_id} receipt=${receipt}"

#!/usr/bin/env bash
# Materialize the two authorized C2 simulator rows (exact16 videos) once.

set -Eeuo pipefail
umask 0027

fail() { echo "[elal3-c2-exact16-materializer] ERROR: $*" >&2; exit 2; }

archive_sha="143e99cfbbafe470f008a3be6cf3a23412ddc0fe3d7e5b41f161c7faa097fce6"
manifest_sha="47a8c1ef2dd1805da91af4eed65868ff668dbec7950449cbeec0bf6814e3f687"
materializer_sha="b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f"
label_sha="1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11"
core_sha="70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862"
train_lora_sha="630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5"
materialize_vae_sha="a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0"
renderer_dataset_sha="afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5"
derivative_sha="543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a"
model_sha="312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d"
contract_sha="92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8"
packet_sha="2c90689dc936ce851f448b23afcd7391af72f9dc8aa4237b887063d1f47c9ecc"

[[ -n "${ELAL3_C2_SOURCE_ARCHIVE:-}" ]] || fail "ELAL3_C2_SOURCE_ARCHIVE is required"
[[ -n "${ELAL3_C2_SOURCE_MANIFEST:-}" ]] || fail "ELAL3_C2_SOURCE_MANIFEST is required"
[[ -n "${ELAL3_C2_OUTPUT_ROOT:-}" ]] || fail "ELAL3_C2_OUTPUT_ROOT is required"
[[ -n "${ELAL3_C2_CONTROLLER_PATH:-}" ]] || fail "ELAL3_C2_CONTROLLER_PATH is required"
[[ "${ELAL3_C2_CONTROLLER_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "ELAL3_C2_CONTROLLER_SHA256 is required"
archive="${ELAL3_C2_SOURCE_ARCHIVE}"
manifest="${ELAL3_C2_SOURCE_MANIFEST}"
output_root="${ELAL3_C2_OUTPUT_ROOT}"
controller_path="${ELAL3_C2_CONTROLLER_PATH}"
controller_sha="${ELAL3_C2_CONTROLLER_SHA256}"

python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
bernini_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
veomni_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
checkpoint_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
experiment_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/elal3_c1_oracle_diagnostic_preflight_20260817_v1"
packet_root="${experiment_root}/simulator_gt_canary_v1"

[[ "${SLURM_JOB_ID:-}:${HOSTNAME%%.*}" == "141620:auh7-1b-gpu-226" ]] || fail "only holder 141620/node226 is authorized"
[[ "${archive}" == /* && "${manifest}" == /* && "${output_root}" == /* && "${controller_path}" == /* ]] || fail "all paths must be absolute"
[[ "${output_root}" != / && "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "unsafe output root"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root is not fresh"
for file in "${archive}" "${manifest}" "${controller_path}" "${python_bin}" "${packet_root}/manifest.json"; do
  [[ -f "${file}" && ! -L "${file}" ]] || fail "missing plain file: ${file}"
done
for directory in "${bernini_root}" "${veomni_root}" "${checkpoint_root}" "${packet_root}"; do
  [[ -d "${directory}" && ! -L "${directory}" ]] || fail "missing directory: ${directory}"
done
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]] || fail "archive SHA differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]] || fail "manifest SHA differs"
[[ "$(sha256sum "${controller_path}" | awk '{print $1}')" == "${controller_sha}" ]] || fail "external launch controller SHA differs"
[[ "$(sha256sum "${packet_root}/manifest.json" | awk '{print $1}')" == "${packet_sha}" ]] || fail "packet SHA differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "scratch parent differs"
scratch="$(mktemp -d "${scratch_parent%/}/elal3-c2-exact16.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  case "${scratch}" in "${scratch_parent%/}/elal3-c2-exact16."*) ;; *) exit 2 ;; esac
  if [[ -d "${scratch}" && ! -L "${scratch}" ]]; then chmod -R u+w -- "${scratch}" || status=2; rm -rf -- "${scratch}" || status=2; fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -- "${scratch}/source"
# C1 established that the holder nodes' default MIOpen sqlite/cache may fail
# with miopenStatusInternalError.  This one-process materializer receives a
# fresh private cache namespace inside the already guarded scratch root.
runtime_cache="${scratch}/runtime-cache"
mkdir -m 0700 -- "${runtime_cache}" \
  "${runtime_cache}/miopen-user" "${runtime_cache}/miopen-custom" \
  "${runtime_cache}/torch" "${runtime_cache}/xdg" "${runtime_cache}/triton"
export MIOPEN_USER_DB_PATH="${runtime_cache}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${runtime_cache}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${runtime_cache}/torch"
export XDG_CACHE_HOME="${runtime_cache}/xdg"
export TRITON_CACHE_DIR="${runtime_cache}/triton"
"${python_bin}" -I -B - "${archive}" "${manifest}" "${scratch}/source" <<'PY'
import hashlib, json, os, stat, sys, tarfile
from pathlib import Path, PurePosixPath

archive, manifest, output = map(Path, sys.argv[1:4])
def reject(message): raise SystemExit("release rejected: " + message)
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
raw_manifest = manifest.read_bytes()
try: value = json.loads(raw_manifest)
except Exception as error: reject("manifest parse failed: " + str(error))
if raw_manifest != canonical(value) + b"\n": reject("manifest is not canonical JSON+newline")
unsigned = dict(value); stored = unsigned.pop("manifest_digest", None)
if stored != hashlib.sha256(canonical(unsigned)).hexdigest(): reject("manifest self digest differs")
if value.get("schema_version") != "bernini-elal3-c2-exact16-materializer-release-v1": reject("schema differs")
raw_archive = archive.read_bytes()
if hashlib.sha256(raw_archive).hexdigest() != value.get("archive_sha256") or len(raw_archive) != value.get("archive_size"): reject("archive binding differs")
if value.get("file_count") != 9 or len(value.get("files", [])) != 9: reject("exact9 closure differs")
if value.get("archive_member_mode") != "0444" or value.get("fresh_runtime_extract_file_mode_required_by_consumer") != "0644" or value.get("fresh_runtime_extract_root_mode") != "0555": reject("mode contract differs")
if any(value.get(key) is not False for key in ("formal_c2_authorized", "exact160_authorized", "source_instruction_inference_authorized", "real_video_generalization_authorized", "scientific_claim_authorized")): reject("forbidden claim authorized")
rows = value["files"]
expected = {row.get("path"): row for row in rows if isinstance(row, dict)}
if len(expected) != 9: reject("duplicate file rows")
with tarfile.open(archive, "r:") as source:
    members = source.getmembers(); names = [row.name for row in members]
    if names != sorted(names, key=lambda item: item.encode("ascii")) or set(names) != set(expected): reject("archive order/closure differs")
    for member in members:
        pure = PurePosixPath(member.name)
        if pure.is_absolute() or ".." in pure.parts or not member.isreg() or member.mode != 0o444 or member.uid != 0 or member.gid != 0 or member.mtime != 0: reject("unsafe member: " + member.name)
        payload = source.extractfile(member).read(); row = expected[member.name]
        if set(row) != {"path", "sha256", "size", "mode"} or row["mode"] != "0444" or len(payload) != row["size"] or hashlib.sha256(payload).hexdigest() != row["sha256"]: reject("member binding differs: " + member.name)
        target = output.joinpath(*pure.parts); target.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            view = memoryview(payload)
            while view:
                count = os.write(fd, view)
                if count <= 0: reject("write stalled")
                view = view[count:]
            os.fchmod(fd, 0o644); os.fsync(fd)
        finally: os.close(fd)
for root, dirs, files in os.walk(output, topdown=False):
    for name in dirs: os.chmod(Path(root) / name, 0o555)
os.chmod(output, 0o555)
for path in output.rglob("*"):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode): reject("symlink after extraction")
    expected_mode = 0o555 if stat.S_ISDIR(info.st_mode) else 0o644
    if stat.S_IMODE(info.st_mode) != expected_mode: reject("runtime extract mode differs")
PY

method_root="${scratch}/source/methods/bernini_action_editing"
materializer="${method_root}/materialize_elal3_simulator_c2_vae_v1.py"
derivative="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_simulator_optimizer_diagnostic_authority_v1.json"
model="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_real_model_authority_v1.json"
contract="${scratch}/source/md/action_editing/20260817_box/evidence/elal3_c2_role_binding_experiment_contract_v1.json"
launcher_path="$(readlink -f "$0")"
launcher_sha="$(sha256sum "${launcher_path}" | awk '{print $1}')"

mkdir -p -- "$(dirname -- "${output_root}")"
mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"
log="${output_root}/materializer.log"
set +e
"${python_bin}" -I -B "${materializer}" \
  --expected-materializer-source-sha256 "${materializer_sha}" --expected-materializer-source-size 50334 \
  --expected-label-source-sha256 "${label_sha}" --expected-label-source-size 76939 \
  --expected-elal3-c0-source-sha256 "${core_sha}" --expected-elal3-c0-source-size 31330 \
  --expected-train-lora-source-sha256 "${train_lora_sha}" --expected-train-lora-source-size 66931 \
  --expected-materialize-vae-source-sha256 "${materialize_vae_sha}" --expected-materialize-vae-source-size 32195 \
  --expected-build-renderer-dataset-source-sha256 "${renderer_dataset_sha}" --expected-build-renderer-dataset-source-size 31012 \
  --bernini-root "${bernini_root}" --veomni-root "${veomni_root}" --checkpoint "${checkpoint_root}" \
  --packet-root "${packet_root}" --packet-manifest-sha256 "${packet_sha}" \
  --derivative-authority "${derivative}" --derivative-authority-sha256 "${derivative_sha}" \
  --model-authority "${model}" --model-authority-sha256 "${model_sha}" \
  --experiment-contract "${contract}" --experiment-contract-sha256 "${contract_sha}" \
  --output "${output_root}/materialized" --ack-simulator-c2-oracle-diagnostic >"${log}" 2>&1
status=$?
set -e
chmod 0444 -- "${log}"
[[ "${status}" -eq 0 ]] || fail "materializer failed with status ${status}; see ${log}"

"${python_bin}" -I -B - \
  "${output_root}" "${archive}" "${manifest}" "${launcher_path}" "${controller_path}" \
  "${archive_sha}" "${manifest_sha}" "${launcher_sha}" "${controller_sha}" "${packet_sha}" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
import torch
from safetensors import safe_open

root, archive, manifest, launcher, controller = map(Path, sys.argv[1:6])
archive_sha, manifest_sha, launcher_sha, controller_sha, packet_sha = sys.argv[6:11]
def reject(message): raise SystemExit("completion rejected: " + message)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
for path, expected in ((archive, archive_sha), (manifest, manifest_sha), (launcher, launcher_sha), (controller, controller_sha)):
    if sha(path) != expected: reject("release/launcher replay differs")
materialized = root / "materialized"; bundle = materialized / "c2-exact16-latents.safetensors"; receipt_path = materialized / "latent-bundle-receipt.json"
if stat.S_IMODE(materialized.stat().st_mode) != 0o555: reject("materialized root mode differs")
for path in (bundle, receipt_path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1: reject("materialized file mode/link differs")
receipt_raw = receipt_path.read_bytes(); receipt = json.loads(receipt_raw)
if receipt_raw != canonical(receipt) + b"\n": reject("receipt is not canonical JSON+newline")
unsigned_receipt = dict(receipt); stored = unsigned_receipt.pop("receipt_digest", None)
if stored != hashlib.sha256(canonical(unsigned_receipt)).hexdigest(): reject("receipt self digest differs")
if receipt.get("schema_version") != "bernini-elal3-simulator-c2-exact16-latent-bundle-receipt-v1": reject("receipt schema differs")
if receipt.get("status") != "ELAL3_SIMULATOR_C2_EXACT16_VAE_GO": reject("receipt status differs")
if receipt.get("packet_binding", {}).get("manifest_file_sha256") != packet_sha: reject("receipt packet binding differs")
rows = receipt.get("tensor_rows")
if not isinstance(rows, list) or len(rows) != 16 or len({row.get("tensor_key") for row in rows}) != 16: reject("receipt tensor closure differs")
expected_keys = [row["tensor_key"] for row in rows]
def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(canonical({"dtype": str(tensor.dtype), "shape": list(map(int, tensor.shape))}) + b"\0")
    view = tensor.view(torch.uint8).reshape(-1)
    for offset in range(0, int(view.numel()), 1 << 20): digest.update(bytes(view[offset:offset + (1 << 20)].tolist()))
    return digest.hexdigest()
with safe_open(bundle, framework="pt", device="cpu") as source:
    keys = list(source.keys())
    if set(keys) != set(expected_keys) or len(keys) != 16: reject("bundle key closure differs")
    for row in rows:
        tensor = source.get_tensor(row["tensor_key"])
        if list(tensor.shape) != [1,16,21,52,70] or str(tensor.dtype) != "torch.float32" or not bool(torch.isfinite(tensor).all().item()): reject("tensor shape/dtype/finite differs")
        if tensor_sha(tensor) != row.get("tensor_sha256"): reject("tensor raw-byte SHA differs")
bundle_sha = sha(bundle); receipt_sha = sha(receipt_path)
bundle_binding = receipt.get("bundle", {})
if bundle_binding.get("sha256") != bundle_sha or bundle_binding.get("size") != bundle.stat().st_size or bundle_binding.get("mode") != 0o444 or bundle_binding.get("nlink") != 1: reject("receipt bundle binding differs")
verification = receipt.get("published_bundle_verification", {})
if verification.get("serialized_payload_sha256") != bundle_sha or verification.get("serialized_payload_size") != bundle.stat().st_size or verification.get("exact16_keys_verified") is not True or verification.get("all_tensors_reloaded_from_serialized_bytes") is not True or len(verification.get("tensor_rows", [])) != 16: reject("published bundle verification differs")
verified_by_key = {row.get("tensor_key"): row for row in verification["tensor_rows"]}
if len(verified_by_key) != 16: reject("published verification duplicate tensor rows")
for row in rows:
    verified = verified_by_key.get(row["tensor_key"], {})
    if verified.get("tensor_sha256") != row.get("tensor_sha256") or verified.get("shape") != [1,16,21,52,70] or verified.get("dtype") != "torch.float32" or verified.get("equals_prewrite_memory_tensor") is not True: reject("published verification tensor row differs")
encoding = receipt.get("encoding", {})
if encoding.get("vae_encode_count") != 16 or any(value is not True for key, value in encoding.items() if key != "vae_encode_count"): reject("encoding replay gates differ")
expected_sources = {
    "materialize_elal3_simulator_c2_vae_v1": ("b9142b3da63499163623248d902c51d01bf5c5295ff171125e3c614dea788c0f", 50334),
    "elal3_simulator_c2_label_v1": ("1f09670a3dd2eae09cd27dbb5fe28c913f618d096a04a64fb0cf1dc9b6e1ec11", 76939),
    "elal3_c0_v1": ("70cd7fe49fda5f25e330d502f33e74bf11407bf892e60c14f70a034f17179862", 31330),
    "train_lora": ("630c215240d4547ea0c347b9fb0bf21324ffe5ee229c5f3673d586a4a0eab4d5", 66931),
    "tools.materialize_vae": ("a09677c9308fa030449f481da42dbc6d1bf2a88cbcd92e9cd1f26c3967bfb6f0", 32195),
    "tools.build_renderer_dataset": ("afc706d4d03bbfee505666476b752cab3fab53a2e80d140de89b06acc162e0f5", 31012),
}
source_binding = receipt.get("runtime_source_bindings", {})
source_rows = source_binding.get("sources", [])
if source_binding.get("source_count") != 6 or len(source_rows) != 6 or source_binding.get("verified_actual_import_module_files") is not True or source_binding.get("verified_before_and_after_encoding") is not True: reject("runtime source closure differs")
for row in source_rows:
    name = row.get("module_name")
    if name not in expected_sources: reject("unexpected runtime source")
    expected_sha, expected_size = expected_sources.pop(name)
    if row.get("sha256") != expected_sha or row.get("size") != expected_size or row.get("mode") != 0o644 or row.get("nlink") != 1 or row.get("held_fd_double_hash_verified") is not True or row.get("held_openat_parent_chain_replayed") is not True or row.get("actual_module_file_verified") is not True: reject("runtime source binding differs")
if expected_sources: reject("missing runtime sources")
packet = receipt.get("packet_binding", {})
if packet.get("exact_media_count") != 16 or len(packet.get("file_rows", [])) != 16 or packet.get("row_ids") != ["c2-three-entity-blocking-response", "c2-three-entity-handover-occlusion"]: reject("live C2 exact16 packet closure differs")
for triple in packet["file_rows"]:
    for kind in ("media", "annotation", "annotation_receipt"):
        binding = triple.get(kind, {})
        if binding.get("mode") != 0o444 or binding.get("nlink") != 1 or binding.get("held_fd_double_read_verified") is not True or binding.get("held_openat_parent_chain_replayed") is not True: reject("live C2 packet held-FD binding differs")
for name, file_sha, object_key, object_value in (
    ("derivative_authority_binding", "543aedd714c7a48c48b4dcc19d1dd6a8bba37d1edda9b1fa195083659380c64a", "authority_digest", "936e91cf3d1d39dd7f45d5f7a4d510dadcbcb4c2f89a8d22581638fccdefd599"),
    ("experiment_contract_binding", "92d700bde0ff9c644f998344d3fecb48bc7c0361f6e948a93c42b924245b25f8", "contract_digest", "18462dcfbeb017e48a7ed6816559667fa8de1911081261cdc103bc6dd9a229d6"),
    ("real_model_authority_binding", "312d74a830ebec675af39b74e31c696ca188068a0e0ac9058745a537961c260d", "authority_digest", "c2c0c9037dea2fd56aa13ac56416bf38c6167686c75b69f0b4b568c82e670c1f"),
):
    binding = receipt.get(name, {})
    held = binding.get("file", {})
    if binding.get("file_sha256") != file_sha or binding.get(object_key) != object_value or binding.get("verified_before_and_after_encoding") is not True or held.get("mode") != 0o644 or held.get("nlink") != 1 or held.get("held_fd_double_read_verified") is not True or held.get("held_openat_parent_chain_replayed") is not True: reject(name + " differs")
model_binding = receipt.get("real_model_authority_binding", {})
if len(model_binding.get("verified_file_bindings", [])) != 9: reject("real model exact9 binding differs")
imported = receipt.get("imported_model_module_bindings", {})
if imported.get("module_count") != 3 or len(imported.get("modules", [])) != 3 or imported.get("verified_before_and_after_encoding") is not True: reject("imported model module closure differs")
scope = receipt.get("authority", {})
if scope.get("teacher_forced_oracle_q_required_for_optimizer_use") is not True or scope.get("training_authority_is_external_and_narrow") is not True or any(scope.get(key) is not False for key in ("formal_c2_authorized", "exact160_authorized", "scientific_claim_authorized", "real_video_data", "source_instruction_inference_authorized", "materializer_source_independently_authorized_here")): reject("receipt authority boundary differs")
run = {
    "schema_version": "bernini-elal3-c2-exact16-materializer-run-complete-v1",
    "status": "COMPLETE_SIMULATOR_C2_EXACT16_ONLY",
    "holder_job_id": "141620", "node": "auh7-1b-gpu-226",
    "release": {"archive_sha256": archive_sha, "archive_size": archive.stat().st_size,
                "manifest_sha256": manifest_sha, "manifest_size": manifest.stat().st_size,
                "launcher_sha256": launcher_sha, "launcher_size": launcher.stat().st_size,
                "external_controller_sha256": controller_sha, "external_controller_size": controller.stat().st_size},
    "packet_manifest_sha256": packet_sha,
    "materialized": {"bundle_relative_path": "materialized/c2-exact16-latents.safetensors",
                     "bundle_sha256": bundle_sha, "bundle_size": bundle.stat().st_size,
                     "receipt_relative_path": "materialized/latent-bundle-receipt.json",
                     "receipt_sha256": receipt_sha, "receipt_size": receipt_path.stat().st_size,
                     "receipt_digest": receipt["receipt_digest"], "tensor_count": 16,
                     "tensor_order": expected_keys},
    "mode_contract": {"archive_member_mode": "0444", "fresh_runtime_extract_file_mode": "0644_required_by_consumer",
                      "fresh_runtime_extract_root_mode": "0555", "published_bundle_and_receipt_mode": "0444"},
    "packet_closure": {"manifest_declared_media_count": 24, "live_c2_media_annotation_receipt_triples_pre_post": 16},
    "materializer_internal_pre_post_final_replay_passed": True,
    "formal_c2_authorized": False, "exact160_authorized": False,
    "source_instruction_inference_authorized": False, "real_video_generalization_authorized": False,
    "scientific_claim_authorized": False,
}
run["run_digest"] = hashlib.sha256(canonical(run)).hexdigest()
payload = canonical(run) + b"\n"; target = root / "RUN_COMPLETE.json"
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
try:
    view = memoryview(payload)
    while view:
        count = os.write(fd, view)
        if count <= 0: reject("RUN_COMPLETE write stalled")
        view = view[count:]
    os.fchmod(fd, 0o444); os.fsync(fd)
finally: os.close(fd)
if target.read_bytes() != payload: reject("RUN_COMPLETE replay differs")
PY
chmod 0555 -- "${output_root}"
echo "[elal3-c2-exact16-materializer] COMPLETE ${output_root}/RUN_COMPLETE.json"

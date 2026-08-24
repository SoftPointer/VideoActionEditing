#!/usr/bin/env bash
# WORLD4 decode of one of the sealed exact-three ELAL-3 C1 simulator oracle-q
# diagnostics on its registered allocation holder.
# This publishes an exact-nine review packet only; it grants no formal-C1,
# source+instruction, real-video, production, exact160, or scientific claim.

set -Eeuo pipefail
umask 0027

fail() {
  echo "[elal3-c1-decode] ERROR: $*" >&2
  exit 2
}

expected_decode_archive_sha256="ee0ab30d9afa17ef5b92b6d0425cbf7c5c0ebaf6cb09e93a6d4165e9021c6119"
expected_decode_manifest_sha256="f850c2470792a9fd0e9c844d1574781d4002abf0f5238690f3842e4f4362b5c2"
expected_decoder_source_sha256="977fa4e6a91d432e57ecaa59dae87419c734d7038fbe92f47369020d65d41c52"
expected_checkpoint_content_manifest_sha256="a95ac2d74fc4379134a6276355d472810ef08e3d9de79761f1244375a6fad831"
expected_training_archive_sha256="631611a96a744025eb6e5b223958908c7dfccfb69bfaefa7432ea9c20afc8194"
expected_training_manifest_sha256="bb56f175f205b626f003c855260243a5c1a5fa3d8c7f0464ddea49931006a9f3"
expected_trainer_source_sha256="521dae4c0f4f7827b021a30cae785a1a8302deb35df96d7ab2411357207005d3"

sha_re='^[0-9a-f]{64}$'
for value in \
  "${expected_decode_archive_sha256}" \
  "${expected_decode_manifest_sha256}" \
  "${expected_decoder_source_sha256}"; do
  [[ "${value}" =~ ${sha_re} ]] || fail "decode release pin is still PENDING"
done

[[ -n "${ELAL3_C1_DECODE_SOURCE_ARCHIVE:-}" ]] || fail "ELAL3_C1_DECODE_SOURCE_ARCHIVE is required"
[[ -n "${ELAL3_C1_DECODE_MANIFEST:-}" ]] || fail "ELAL3_C1_DECODE_MANIFEST is required"
[[ -n "${ELAL3_C1_DECODE_OUTPUT_ROOT:-}" ]] || fail "ELAL3_C1_DECODE_OUTPUT_ROOT is required"
[[ -n "${ELAL3_C1_DECODE_LAUNCHER_SHA256:-}" ]] || fail "ELAL3_C1_DECODE_LAUNCHER_SHA256 is required"
decode_archive="${ELAL3_C1_DECODE_SOURCE_ARCHIVE}"
decode_manifest="${ELAL3_C1_DECODE_MANIFEST}"
output_root="${ELAL3_C1_DECODE_OUTPUT_ROOT}"
expected_launcher_sha256="${ELAL3_C1_DECODE_LAUNCHER_SHA256}"
launcher_path="$(realpath -- "$0")"

python_bin="/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
bernini_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/motive_action_repr_auto/vendor/Bernini-2d2b4591"
veomni_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_r13_action_81f_v1/vendor/VeOmni-v0.1.11"
checkpoint_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit/checkpoints/Bernini-R-1.3B-Diffusers-ff4c5d4"
experiment_root="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/elal3_c1_oracle_diagnostic_preflight_20260817_v1"
training_release_root="${experiment_root}/c1-oracle-train-release-r3"
training_archive="${training_release_root}/source.tar"
training_manifest="${training_release_root}/source.manifest.json"
packet_root="${experiment_root}/simulator_gt_canary_v1"
latent_bundle="${experiment_root}/vae-c1-row-modelbound-v2/c1-latents.safetensors"
world4_log="${output_root}.world4.log"

job_id="${SLURM_JOB_ID:-}"
node="$(hostname -s)"
case "${job_id}:${node}" in
  141620:auh7-1b-gpu-226)
    sampling_seed="20260817"
    training_run_name="elal3_c1_node226_seed20260817_r3/elal3_c1_ten_step_overfit"
    registered_output_name="elal3_c1_node226_seed20260817_r3_decode_r1"
    expected_training_receipt_sha256="a7ca5e4ec2fd04ccd77bfd943bee48cb4978561787a62f5d21175d9846b3af71"
    expected_step0_adapter_sha256="0369c6dd3dfa5b58e2eb67984955babe4ab637edef1d50a7eb60628b07be1f38"
    expected_trained_adapter_sha256="c38ba270b0ff2736c06ec4733b1b9bf4858a7654adb0f020b55a98a406282ac9"
    ;;
  141618:auh7-1b-gpu-249)
    sampling_seed="20260818"
    training_run_name="elal3_c1_node249_seed20260818_r3/elal3_c1_ten_step_overfit"
    registered_output_name="elal3_c1_node249_seed20260818_r3_decode_r1"
    expected_training_receipt_sha256="37b51f4f0003e0e4418664906106dd2ad25b5a09b1be866df3eeac0e0f3362d8"
    expected_step0_adapter_sha256="96158cb165f1f3c0d151c27f79bcb71439cbd42a024e04e30b94214428d33dbb"
    expected_trained_adapter_sha256="1108680084e976904c6c33586af556327100cfd94bb0d3891212800a9b0dea69"
    ;;
  141619:auh7-1b-gpu-257)
    sampling_seed="20260819"
    training_run_name="elal3_c1_node257_seed20260819_r3/elal3_c1_ten_step_overfit"
    registered_output_name="elal3_c1_node257_seed20260819_r3_decode_r1"
    expected_training_receipt_sha256="a67aa4b7235ad130cdb20b4060865fd9014a0c10437828cc1d3bc0b8a6eccb7c"
    expected_step0_adapter_sha256="6a0abffed80bcf3d5021a05dbb080a8c39e785076e1fb162d88a5ffad8ddb4cd"
    expected_trained_adapter_sha256="888f14297cbac3523cd0eb1ccd53892118e739692a5a4319a5ce1d4dd35be4d9"
    ;;
  *)
    fail "decoder requires one registered exact-three allocation holder"
    ;;
esac
training_run="${experiment_root}/${training_run_name}"
registered_output_root="${experiment_root}/${registered_output_name}"

[[ "${decode_archive}" == /* && "${decode_manifest}" == /* && "${output_root}" == /* ]] || fail "decode paths must be absolute"
[[ "${output_root}" == "${registered_output_root}" ]] || fail "decode output is not the registered fresh path"
[[ "${expected_launcher_sha256}" =~ ${sha_re} ]] || fail "expected launcher SHA-256 is invalid"
[[ "${launcher_path}" == /* && -f "${launcher_path}" && ! -L "${launcher_path}" ]] || fail "launcher path is not a plain absolute file"
[[ ! -e "${output_root}" && ! -L "${output_root}" && ! -e "${world4_log}" && ! -L "${world4_log}" ]] || fail "decode output/log must be fresh"
for path in "${decode_archive}" "${decode_manifest}" "${training_archive}" "${training_manifest}" "${python_bin}" "${latent_bundle}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "required plain file unavailable: ${path}"
done
for path in "${bernini_root}" "${veomni_root}" "${checkpoint_root}" "${packet_root}" "${training_run}"; do
  [[ -d "${path}" && ! -L "${path}" ]] || fail "required directory unavailable: ${path}"
done
[[ "$(sha256sum "${launcher_path}" | awk '{print $1}')" == "${expected_launcher_sha256}" ]] || fail "decode launcher SHA-256 differs"
[[ "$(sha256sum "${decode_archive}" | awk '{print $1}')" == "${expected_decode_archive_sha256}" ]] || fail "decode archive SHA-256 differs"
[[ "$(sha256sum "${decode_manifest}" | awk '{print $1}')" == "${expected_decode_manifest_sha256}" ]] || fail "decode manifest SHA-256 differs"
[[ "$(sha256sum "${training_archive}" | awk '{print $1}')" == "${expected_training_archive_sha256}" ]] || fail "training archive SHA-256 differs"
[[ "$(sha256sum "${training_manifest}" | awk '{print $1}')" == "${expected_training_manifest_sha256}" ]] || fail "training manifest SHA-256 differs"
[[ "$(sha256sum "${training_run}/TRAINING_RECEIPT.json" | awk '{print $1}')" == "${expected_training_receipt_sha256}" ]] || fail "training receipt SHA-256 differs"
[[ "$(sha256sum "${training_run}/checkpoints/checkpoint-00000000/adapter-and-elal3.pt" | awk '{print $1}')" == "${expected_step0_adapter_sha256}" ]] || fail "step0 adapter SHA-256 differs"
[[ "$(sha256sum "${training_run}/checkpoints/checkpoint-00000010/adapter-and-elal3.pt" | awk '{print $1}')" == "${expected_trained_adapter_sha256}" ]] || fail "trained adapter SHA-256 differs"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
scratch="$(mktemp -d "${scratch_parent%/}/elal3-c1-decode.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  case "${scratch}" in "${scratch_parent%/}/elal3-c1-decode."*) ;; *) exit 2 ;; esac
  if [[ -d "${scratch}" && ! -L "${scratch}" ]]; then
    chmod -R u+w -- "${scratch}" || status=2
    rm -rf -- "${scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
mkdir -m 0700 -- "${scratch}/training" "${scratch}/decode" "${scratch}/rank-cache"

# Extract both reviewed USTARs without trusting pathnames or archive metadata.
"${python_bin}" -I -B - \
  "${training_archive}" "${training_manifest}" "${scratch}/training" training \
  "${decode_archive}" "${decode_manifest}" "${scratch}/decode" decode \
  "${expected_decoder_source_sha256}" \
  "${expected_checkpoint_content_manifest_sha256}" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile

training_archive, training_manifest, training_dest = map(Path, sys.argv[1:4])
decode_archive, decode_manifest, decode_dest = map(Path, sys.argv[5:8])
expected_decoder = sys.argv[9]
expected_checkpoint_manifest = sys.argv[10]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")

def extract(archive_path, manifest_path, destination, kind):
    raw = manifest_path.read_bytes()
    value = json.loads(raw)
    unsigned = dict(value)
    stored = unsigned.pop("manifest_digest", None)
    if raw != canonical(value) + b"\n" or stored != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise SystemExit(kind + " manifest canonical digest differs")
    rows = value.get("files")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(kind + " manifest files differ")
    expected = {row.get("path"): row for row in rows}
    if len(expected) != len(rows) or None in expected:
        raise SystemExit(kind + " manifest file closure differs")
    if value.get("archive_sha256") != hashlib.sha256(archive_path.read_bytes()).hexdigest():
        raise SystemExit(kind + " manifest/archive binding differs")
    with tarfile.open(archive_path, "r:") as archive:
        members = archive.getmembers()
        if [member.name for member in members] != list(expected):
            raise SystemExit(kind + " USTAR order/closure differs")
        for member in members:
            pure = PurePosixPath(member.name)
            row = expected[member.name]
            mode = row.get("archive_mode") if kind == "decode" else int(row.get("mode", "0"), 8)
            if (
                pure.is_absolute() or ".." in pure.parts or not member.isreg()
                or member.mode != mode or member.uid != 0 or member.gid != 0
                or member.mtime != 0 or member.size != row.get("size")
            ):
                raise SystemExit(kind + " unsafe USTAR member: " + member.name)
            source = archive.extractfile(member)
            if source is None:
                raise SystemExit(kind + " member payload absent")
            payload = source.read()
            if len(payload) != member.size or hashlib.sha256(payload).hexdigest() != row.get("sha256"):
                raise SystemExit(kind + " member payload differs: " + member.name)
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise SystemExit("extract write made no progress")
                    remaining = remaining[written:]
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return value

extract(training_archive, training_manifest, training_dest, "training")
decode = extract(decode_archive, decode_manifest, decode_dest, "decode")
member = "methods/bernini_action_editing/decode_elal3_c1_simulator_oracle_q_v1.py"
if decode.get("decoder_member") != member or decode.get("decoder_source_sha256") != expected_decoder:
    raise SystemExit("decode source manifest binding differs")
checkpoint_member = "methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
if decode.get("checkpoint_content_manifest") != {
    "member": checkpoint_member,
    "sha256": expected_checkpoint_manifest,
    "size": 2350,
    "row_count": 23,
}:
    raise SystemExit("checkpoint content manifest release binding differs")
PY

decoder="${scratch}/decode/methods/bernini_action_editing/decode_elal3_c1_simulator_oracle_q_v1.py"
checkpoint_content_manifest="${scratch}/decode/methods/bernini_action_editing/audits/bernini_r13_ff4c5d4_checkpoint.sha256"
[[ "$(sha256sum "${decoder}" | awk '{print $1}')" == "${expected_decoder_source_sha256}" ]] || fail "extracted decoder SHA-256 differs"
[[ "$(sha256sum "${checkpoint_content_manifest}" | awk '{print $1}')" == "${expected_checkpoint_content_manifest_sha256}" ]] || fail "extracted checkpoint content manifest SHA-256 differs"

rank_wrapper="${scratch}/rank_exec.sh"
"${python_bin}" -I -B - "${rank_wrapper}" <<'PY'
import os
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = b'''#!/usr/bin/env bash
set -Eeuo pipefail
umask 0077
cache_base="${1:?rank cache base missing}"
shift
rank="${LOCAL_RANK:?LOCAL_RANK missing}"
[[ "${rank}" =~ ^[0-3]$ ]] || { echo "invalid LOCAL_RANK=${rank}" >&2; exit 2; }
rank_root="${cache_base}/rank_${rank}"
[[ ! -e "${rank_root}" && ! -L "${rank_root}" ]] || { echo "rank cache exists: ${rank_root}" >&2; exit 2; }
mkdir -m 0700 -- "${rank_root}" "${rank_root}/miopen-user" "${rank_root}/miopen-custom" "${rank_root}/torch" "${rank_root}/xdg" "${rank_root}/triton"
export MIOPEN_USER_DB_PATH="${rank_root}/miopen-user"
export MIOPEN_CUSTOM_CACHE_DIR="${rank_root}/miopen-custom"
export TORCH_EXTENSIONS_DIR="${rank_root}/torch"
export XDG_CACHE_HOME="${rank_root}/xdg"
export TRITON_CACHE_DIR="${rank_root}/triton"
exec "$@"
'''
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o500)
try:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SystemExit("rank wrapper write made no progress")
        remaining = remaining[written:]
    os.fchmod(descriptor, 0o500)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

set +e
"${python_bin}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --master-port=29621 --no-python \
  "${rank_wrapper}" "${scratch}/rank-cache" "${python_bin}" -B "${decoder}" \
    --release-root "${scratch}/training" \
    --release-manifest "${training_manifest}" \
    --expected-release-manifest-sha256 "${expected_training_manifest_sha256}" \
    --expected-decoder-source-sha256 "${expected_decoder_source_sha256}" \
    --expected-trainer-source-sha256 "${expected_trainer_source_sha256}" \
    --checkpoint-content-manifest "${checkpoint_content_manifest}" \
    --expected-checkpoint-content-manifest-sha256 "${expected_checkpoint_content_manifest_sha256}" \
    --decode-release-manifest "${decode_manifest}" \
    --expected-decode-release-manifest-sha256 "${expected_decode_manifest_sha256}" \
    --decode-launcher "${launcher_path}" \
    --expected-decode-launcher-sha256 "${expected_launcher_sha256}" \
    --bernini-root "${bernini_root}" \
    --veomni-root "${veomni_root}" \
    --checkpoint "${checkpoint_root}" \
    --packet-root "${packet_root}" \
    --latent-bundle "${latent_bundle}" \
    --training-run "${training_run}" \
    --expected-training-receipt-sha256 "${expected_training_receipt_sha256}" \
    --expected-step0-adapter-sha256 "${expected_step0_adapter_sha256}" \
    --expected-trained-adapter-sha256 "${expected_trained_adapter_sha256}" \
    --output "${output_root}" \
    --sampling-seed "${sampling_seed}" \
    --num-inference-steps 40 \
    --ack-simulator-oracle-q-only \
    --ack-not-source-instruction-inference \
    --ack-not-formal-c1 \
    --ack-no-scientific-claim \
    >"${world4_log}" 2>&1
decode_status=$?
set -e
chmod 0444 -- "${world4_log}"
if [[ "${decode_status}" -ne 0 ]]; then
  if [[ -d "${output_root}" && ! -L "${output_root}" ]]; then chmod -R a-w -- "${output_root}" || true; fi
  fail "WORLD4 decoder failed with status ${decode_status}; log sealed"
fi

# Decoder-native receipt validation is finalized together with the reviewed
# decoder source; these checks intentionally fail if backlink fields are absent.
"${python_bin}" -I -B - \
  "${output_root}" "${decode_manifest}" "${expected_decode_manifest_sha256}" \
  "${launcher_path}" "${expected_launcher_sha256}" \
  "${checkpoint_content_manifest}" \
  "${expected_checkpoint_content_manifest_sha256}" \
  "${checkpoint_root}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
manifest_path = Path(sys.argv[2]).resolve(strict=True)
manifest_sha = sys.argv[3]
launcher_path = Path(sys.argv[4]).resolve(strict=True)
launcher_sha = sys.argv[5]
checkpoint_manifest_path = Path(sys.argv[6]).resolve(strict=True)
checkpoint_manifest_sha = sys.argv[7]
checkpoint_root = Path(sys.argv[8]).resolve(strict=True)
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
is_sha = lambda value: isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")

def verify_plain_digest(path, expected, label, expected_size=None):
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SystemExit(label + " is not a plain absolute file")
    before = path.stat()
    if (
        before.st_nlink != 1
        or not 0 < before.st_size <= (1 << 20)
        or (expected_size is not None and before.st_size != expected_size)
    ):
        raise SystemExit(label + " file metadata differs")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != before.st_size
        or hashlib.sha256(payload).hexdigest() != expected
    ):
        raise SystemExit(label + " terminal digest differs")

verify_plain_digest(manifest_path, manifest_sha, "decode manifest")
verify_plain_digest(launcher_path, launcher_sha, "decode launcher")
verify_plain_digest(
    checkpoint_manifest_path,
    checkpoint_manifest_sha,
    "checkpoint content manifest",
    expected_size=2350,
)
receipt_path = root / "DECODE_RECEIPT.json"
raw = receipt_path.read_bytes()
receipt = json.loads(raw)
unsigned = dict(receipt)
stored = unsigned.pop("receipt_digest", None)
if raw != canonical(receipt) + b"\n" or stored != hashlib.sha256(canonical(unsigned)).hexdigest():
    raise SystemExit("decode receipt canonical self-digest differs")
if (
    receipt.get("schema_version")
    != "bernini-elal3-c1-simulator-oracle-q-decode-receipt-v3"
    or receipt.get("method")
    != "bernini-elal3-c1-simulator-oracle-q-checkpoint-decode-v3"
    or receipt.get("status") != "SIMULATOR_ORACLE_Q_EXACT9_REVIEW_READY"
    or receipt.get("elal_branches_teacher_forced_simulator_oracle_q") is not True
    or receipt.get("frozen_base_has_no_elal_q_input") is not True
    or receipt.get("source_instruction_inference") is not False
    or receipt.get("formal_c1_authorized") is not False
    or receipt.get("exact160_authorized") is not False
    or receipt.get("real_video_data") is not False
    or receipt.get("scientific_claim_authorized") is not False
    or receipt.get("action_encoder_qualified") is not False
):
    raise SystemExit("decode receipt schema/status/scope differs")
decode_release = receipt.get("decode_release")
if (
    not isinstance(decode_release, dict)
    or decode_release.get("manifest_path") != str(manifest_path)
    or decode_release.get("manifest_sha256") != manifest_sha
    or decode_release.get("launcher_path") != str(launcher_path)
    or decode_release.get("launcher_sha256") != launcher_sha
):
    raise SystemExit("decode receipt release/launcher backlink differs")
checkpoint_authority = receipt.get("checkpoint_content_authority")
if (
    not isinstance(checkpoint_authority, dict)
    or set(checkpoint_authority)
    != {
        "manifest_path",
        "manifest_sha256",
        "manifest_size",
        "row_count",
        "ordered_manifest_rows_sha256",
        "pre_load_world4_replay",
        "final_pre_publish_world4_replay",
        "exact23_unchanged_across_runtime",
    }
    or checkpoint_authority.get("manifest_path") != str(checkpoint_manifest_path)
    or checkpoint_authority.get("manifest_sha256") != checkpoint_manifest_sha
    or checkpoint_authority.get("manifest_size") != 2350
    or checkpoint_authority.get("row_count") != 23
    or checkpoint_authority.get("exact23_unchanged_across_runtime") is not True
):
    raise SystemExit("decode checkpoint content authority differs")
ordered_rows_sha = checkpoint_authority.get("ordered_manifest_rows_sha256")
if not is_sha(ordered_rows_sha):
    raise SystemExit("decode checkpoint manifest rows digest differs")
replays = (
    (
        checkpoint_authority.get("pre_load_world4_replay"),
        "decoder_checkpoint_pre_load",
    ),
    (
        checkpoint_authority.get("final_pre_publish_world4_replay"),
        "decoder_checkpoint_final_pre_publish",
    ),
)
content_rows_sha = None
for replay, stage in replays:
    if (
        not isinstance(replay, dict)
        or set(replay)
        != {
            "stage",
            "checkpoint_root",
            "checkpoint_content_manifest_sha256",
            "row_count",
            "content_rows_sha256",
            "exact23_full_stable_rehash_by_rank_zero",
            "world_size",
            "world4_broadcast_identity_verified",
            "world4_rank_receipt_digest_consensus",
            "ordered_world4_rank_receipt_digests",
        }
        or replay.get("stage") != stage
        or replay.get("checkpoint_root") != str(checkpoint_root)
        or replay.get("checkpoint_content_manifest_sha256")
        != checkpoint_manifest_sha
        or replay.get("row_count") != 23
        or replay.get("exact23_full_stable_rehash_by_rank_zero") is not True
        or replay.get("world_size") != 4
        or replay.get("world4_broadcast_identity_verified") is not True
        or replay.get("world4_rank_receipt_digest_consensus") is not True
    ):
        raise SystemExit("decode checkpoint content replay differs")
    replay_rows_sha = replay.get("content_rows_sha256")
    rank_digests = replay.get("ordered_world4_rank_receipt_digests")
    if (
        not is_sha(replay_rows_sha)
        or not isinstance(rank_digests, list)
        or len(rank_digests) != 4
        or any(
            not is_sha(digest)
            for digest in rank_digests
        )
        or len(set(rank_digests)) != 1
    ):
        raise SystemExit("decode checkpoint replay digest consensus differs")
    if content_rows_sha is None:
        content_rows_sha = replay_rows_sha
    elif replay_rows_sha != content_rows_sha:
        raise SystemExit("decode checkpoint content changed across runtime")
order = ["source", "gt_target", "appearance_anchor", "frozen_base", "step0_correct_q", "trained_correct_q", "trained_zero_q", "trained_phase_reverse_q", "trained_role_swap_q"]
media = receipt.get("media")
if not isinstance(media, list) or [row.get("key") for row in media] != order:
    raise SystemExit("decode exact9 order differs")
expected_entries = {"DECODE_RECEIPT.json", "index.html"}
for index, row in enumerate(media):
    relative = row.get("relative_path")
    if relative != f"{index:02d}_{order[index]}.mp4":
        raise SystemExit("decode media filename differs")
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.stat().st_size != row.get("size"):
        raise SystemExit("decode media file differs")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row.get("sha256") or row.get("frame_count") != 81 or row.get("fps") != 25.0:
        raise SystemExit("decode media attestation differs")
    expected_entries.add(relative)
html = receipt.get("html")
html_path = root / "index.html"
if not isinstance(html, dict) or html.get("relative_path") != "index.html" or hashlib.sha256(html_path.read_bytes()).hexdigest() != html.get("sha256"):
    raise SystemExit("decode HTML attestation differs")
if {path.name for path in root.iterdir()} != expected_entries:
    raise SystemExit("decode output closure differs")
PY

chmod -R a-w -- "${output_root}"
chmod 0555 -- "${output_root}"
echo "[elal3-c1-decode] COMPLETE ${job_id} ${node} output=${output_root}"

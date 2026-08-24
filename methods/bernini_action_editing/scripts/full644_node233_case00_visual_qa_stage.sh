#!/bin/bash
set -euo pipefail
umask 077

# Local, read-only staging and decode QA for the one fresh node233 r5c case00
# canary.  This script never starts a Slurm step and never writes remotely.

REMOTE_HOST="auh"
REMOTE_ROOT="/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit_experiments/bernini_full644_exploratory_matched_eval_auh_r5c_job143808_node233_case00_847b91a2_c91de7eb_cb201398_r1"
SOURCE_PATH="/vast/users/guangyi.chen/dataset/goku/subject_movement/extracted/videos/1852ada01d7c43a4/source.mp4"
PLAN_SHA256="498a08481e6ced30d885f45aa8184ca214c77c0330d793c8b24702e8e999184e"
PLAN_DIGEST="5b0f948f88f28fb46ba35ca6451a63b6dd334902ba5abf6d98dc92b4b6f3ca68"
SOURCE_SHA256="84d8361bb53d9a210b5c19ceba22ac31ba7a3b008760afd132f865065266bbf7"
INSTRUCTION_SHA256="736188e6b5dfdbe06e82132caf94427745bf4d39c1b76a3f4385fe81e11ab5f3"
ADAPTER_MODEL_SHA256="44efdc5a0501238250b1d32ae2859abe248ffc37b152cd8db86ff84b378d6b22"

for executable in ssh scp jq ffprobe ffmpeg shasum mktemp; do
  command -v "$executable" >/dev/null 2>&1 || {
    printf 'missing required executable: %s\n' "$executable" >&2
    exit 1
  }
done

# This is a read-only login-node preflight.  The positional root is fixed above;
# no caller-supplied remote path can redirect the audit to an older campaign.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" /bin/bash -s -- "$REMOTE_ROOT" "$SOURCE_PATH" <<'REMOTE_PREFLIGHT'
set -euo pipefail
root="$1"
source_path="$2"
plan="$root/plan/full644_exploratory_matched_plan_auh_r5c.json"
evidence="$root/logs/case00_gpu_canary.sacct-and-replay.json"
report="$root/final/case00_canary_nonformal_report_auh_r5c.json"
attestation="$root/final/case00_canary_runner_attestation_auh_r5c.json"
base="$root/outputs/media/case00-base.mp4"
base_receipt="$base.receipt.json"
adapted="$root/outputs/media/case00-full644.mp4"
adapted_receipt="$adapted.receipt.json"

expected_media=$'case00-base.mp4\ncase00-base.mp4.receipt.json\ncase00-full644.mp4\ncase00-full644.mp4.receipt.json'
actual_media="$(find "$root/outputs/media" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)"
[[ "$actual_media" == "$expected_media" ]] || {
  printf '%s\n' 'remote media closure is not the exact fresh case00 pair' >&2
  exit 10
}
expected_final=$'case00_canary_nonformal_report_auh_r5c.json\ncase00_canary_runner_attestation_auh_r5c.json'
actual_final="$(find "$root/final" -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | LC_ALL=C sort)"
[[ "$actual_final" == "$expected_final" ]] || {
  printf '%s\n' 'remote final closure is not the exact case00 pair' >&2
  exit 11
}

[[ "$(sha256sum "$plan" | cut -d' ' -f1)" == "498a08481e6ced30d885f45aa8184ca214c77c0330d793c8b24702e8e999184e" ]]
[[ "$(sha256sum "$source_path" | cut -d' ' -f1)" == "84d8361bb53d9a210b5c19ceba22ac31ba7a3b008760afd132f865065266bbf7" ]]
[[ "$(stat -c '%a|%h' "$evidence")" == '400|1' ]]
[[ "$(stat -c '%a|%h' "$report")" == '444|1' ]]
[[ "$(stat -c '%a|%h' "$attestation")" == '444|1' ]]
[[ "$(stat -c '%a|%h' "$base")" == '444|1' ]]
[[ "$(stat -c '%a|%h' "$adapted")" == '444|1' ]]
[[ "$(stat -c '%a|%h' "$base_receipt")" == '400|1' ]]
[[ "$(stat -c '%a|%h' "$adapted_receipt")" == '400|1' ]]

jq -e '
  .schema_version == "full644-r5c-case00-gpu-heldfd-controller-evidence-v3" and
  .status == "PASS" and
  .campaign_mode == "case00-pair-canary" and
  .holder_job_id == "143808" and
  .node == "auh7-1b-gpu-233" and
  .single_srun_attempt == true and
  .srun_returncode == 0 and
  .manual_visual_review_required == true and
  .visual_review_performed == false and
  .full16_authorized == false and
  .html_generated == false and
  ([.media[].task_id] == ["shared8-00-base", "shared8-00-full644"])
' "$evidence" >/dev/null
numeric_step="$(jq -r '.numeric_step' "$evidence")"
[[ "$numeric_step" =~ ^143808\.([1-9][0-9]*)$ ]]
(( BASH_REMATCH[1] > 144 ))

[[ "$(sha256sum "$report" | cut -d' ' -f1)" == "$(jq -r '.report.sha256' "$evidence")" ]]
[[ "$(sha256sum "$attestation" | cut -d' ' -f1)" == "$(jq -r '.attestation.sha256' "$evidence")" ]]
[[ "$(sha256sum "$base" | cut -d' ' -f1)" == "$(jq -r '.media[] | select(.task_id == "shared8-00-base") | .video_sha256' "$evidence")" ]]
[[ "$(sha256sum "$base_receipt" | cut -d' ' -f1)" == "$(jq -r '.media[] | select(.task_id == "shared8-00-base") | .receipt_sha256' "$evidence")" ]]
[[ "$(sha256sum "$adapted" | cut -d' ' -f1)" == "$(jq -r '.media[] | select(.task_id == "shared8-00-full644") | .video_sha256' "$evidence")" ]]
[[ "$(sha256sum "$adapted_receipt" | cut -d' ' -f1)" == "$(jq -r '.media[] | select(.task_id == "shared8-00-full644") | .receipt_sha256' "$evidence")" ]]
REMOTE_PREFLIGHT

stage="$(mktemp -d /tmp/full644-node233-case00-visual-qa.XXXXXX)"
mkdir -p "$stage/raw" "$stage/probe" "$stage/frames" "$stage/contact_sheets/key5" "$stage/contact_sheets/all81"

copy_one() {
  remote_path="$1"
  local_name="$2"
  scp -q "$REMOTE_HOST:$remote_path" "$stage/raw/$local_name"
}

copy_one "$REMOTE_ROOT/plan/full644_exploratory_matched_plan_auh_r5c.json" plan.json
copy_one "$REMOTE_ROOT/logs/case00_gpu_canary.sacct-and-replay.json" controller_evidence.json
copy_one "$REMOTE_ROOT/final/case00_canary_nonformal_report_auh_r5c.json" report.json
copy_one "$REMOTE_ROOT/final/case00_canary_runner_attestation_auh_r5c.json" attestation.json
copy_one "$REMOTE_ROOT/outputs/media/case00-base.mp4" case00-base.mp4
copy_one "$REMOTE_ROOT/outputs/media/case00-base.mp4.receipt.json" case00-base.receipt.json
copy_one "$REMOTE_ROOT/outputs/media/case00-full644.mp4" case00-full644.mp4
copy_one "$REMOTE_ROOT/outputs/media/case00-full644.mp4.receipt.json" case00-full644.receipt.json
copy_one "$SOURCE_PATH" case00-source.mp4

/usr/bin/python3 - "$stage" "$REMOTE_ROOT" "$SOURCE_PATH" "$PLAN_SHA256" "$PLAN_DIGEST" "$SOURCE_SHA256" "$INSTRUCTION_SHA256" "$ADAPTER_MODEL_SHA256" "$(command -v ffprobe)" <<'PY_VALIDATE'
import hashlib
import json
import pathlib
import subprocess
import sys

(
    stage_arg,
    remote_root,
    source_path,
    expected_plan_sha,
    expected_plan_digest,
    expected_source_sha,
    expected_instruction_sha,
    expected_adapter_sha,
    ffprobe,
) = sys.argv[1:]
stage = pathlib.Path(stage_arg)
raw_root = stage / "raw"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_canonical(name):
    path = raw_root / name
    raw = path.read_bytes()
    value = json.loads(
        raw.decode("utf-8", "strict"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if type(value) is not dict or raw != canonical:
        raise RuntimeError(f"non-canonical JSON: {name}")
    return value


def embedded_digest(value, field):
    body = dict(value)
    claimed = body.pop(field, None)
    actual = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if claimed != actual:
        raise RuntimeError(f"embedded digest mismatch: {field}")
    return claimed


plan = load_canonical("plan.json")
evidence = load_canonical("controller_evidence.json")
report = load_canonical("report.json")
attestation = load_canonical("attestation.json")
base_receipt = load_canonical("case00-base.receipt.json")
adapted_receipt = load_canonical("case00-full644.receipt.json")

if sha256(raw_root / "plan.json") != expected_plan_sha:
    raise RuntimeError("local plan bytes differ")
if embedded_digest(plan, "plan_digest") != expected_plan_digest:
    raise RuntimeError("local plan digest differs")
evidence_digest = embedded_digest(evidence, "evidence_digest")
report_digest = embedded_digest(report, "report_digest")
attestation_digest = embedded_digest(attestation, "attestation_digest")
base_receipt_digest = embedded_digest(base_receipt, "receipt_digest")
adapted_receipt_digest = embedded_digest(adapted_receipt, "receipt_digest")

selected = ["shared8-00-base", "shared8-00-full644"]
if (
    evidence.get("status") != "PASS"
    or evidence.get("node") != "auh7-1b-gpu-233"
    or evidence.get("media") is None
    or [row.get("task_id") for row in evidence["media"]] != selected
    or evidence.get("html_generated") is not False
    or evidence.get("full16_authorized") is not False
):
    raise RuntimeError("controller evidence closure differs")
if (
    report.get("status") != "CANARY_COMPLETE_AWAITING_VISUAL_REVIEW"
    or report.get("selected_task_ids") != selected
    or report.get("verified_task_count") != 2
    or report.get("html_generated") is not False
    or report.get("formal_full16_report") is not False
    or [row.get("task_id") for row in report.get("results", [])] != selected
):
    raise RuntimeError("case00 report closure differs")
if (
    attestation.get("status") != "CANARY_COMPLETE_AWAITING_VISUAL_REVIEW"
    or attestation.get("task_ids") != selected
    or attestation.get("task_count") != 2
    or attestation.get("retry_count") != 0
    or attestation.get("all_selected_tasks_attempted_exactly_once") is not True
    or attestation.get("all_selected_tasks_succeeded") is not True
    or attestation.get("formal_full16_report") is not False
):
    raise RuntimeError("case00 attestation closure differs")
if (
    evidence["report"].get("sha256") != sha256(raw_root / "report.json")
    or evidence["report"].get("report_digest") != report_digest
    or evidence["attestation"].get("sha256") != sha256(raw_root / "attestation.json")
    or evidence["attestation"].get("attestation_digest") != attestation_digest
    or attestation.get("verified_report", {}).get("sha256") != sha256(raw_root / "report.json")
    or attestation.get("verified_report", {}).get("report_digest") != report_digest
):
    raise RuntimeError("report/attestation cross-link differs")

case_tasks = plan.get("tasks", [])[:2]
if [task.get("task_id") for task in case_tasks] != selected:
    raise RuntimeError("plan case00 order differs")
for task in case_tasks:
    if (
        task.get("iid") != "1852ada01d7c43a4"
        or task.get("instruction_sha256") != expected_instruction_sha
        or task.get("instruction")
        != "Show the car driving dynamically through the snowy landscape, kicking up snow."
        or task.get("seed") != 2026
        or task.get("num_inference_steps") != 40
        or task.get("source_video") != source_path
        or task.get("source_video_sha256") != expected_source_sha
    ):
        raise RuntimeError("plan case00 task tuple differs")

if sha256(raw_root / "case00-source.mp4") != expected_source_sha:
    raise RuntimeError("staged source bytes differ")

receipts = {
    "shared8-00-base": (base_receipt, base_receipt_digest, raw_root / "case00-base.mp4", raw_root / "case00-base.receipt.json"),
    "shared8-00-full644": (adapted_receipt, adapted_receipt_digest, raw_root / "case00-full644.mp4", raw_root / "case00-full644.receipt.json"),
}
for task in case_tasks:
    task_id = task["task_id"]
    receipt, receipt_digest, video_path, receipt_path = receipts[task_id]
    media = next(row for row in evidence["media"] if row["task_id"] == task_id)
    result = next(row for row in report["results"] if row["task_id"] == task_id)
    video_sha = sha256(video_path)
    receipt_sha = sha256(receipt_path)
    if (
        media.get("video_sha256") != video_sha
        or media.get("receipt_sha256") != receipt_sha
        or media.get("receipt_digest") != receipt_digest
        or result.get("output_sha256") != video_sha
        or result.get("receipt_file_sha256") != receipt_sha
        or result.get("receipt_digest") != receipt_digest
        or receipt.get("input", {}).get("source_video_sha256") != expected_source_sha
        or receipt.get("input", {}).get("instruction_utf8_sha256") != expected_instruction_sha
        or receipt.get("sampling", {}).get("seed") != 2026
        or receipt.get("sampling", {}).get("num_frames") != 81
        or receipt.get("sampling", {}).get("num_inference_steps") != 40
        or receipt.get("preprocessing", {}).get("frame_count") != 81
        or receipt.get("preprocessing", {}).get("fps") != 25.0
        or receipt.get("output", {}).get("sha256") != video_sha
        or receipt.get("output", {}).get("size") != video_path.stat().st_size
        or receipt.get("output", {}).get("frame_count") != 81
        or receipt.get("output", {}).get("fps") != 25.0
        or receipt.get("output", {}).get("audio_preserved") is not False
    ):
        raise RuntimeError(f"staged media/receipt closure differs: {task_id}")

if (
    base_receipt.get("adapter", {}).get("enabled") is not False
    or base_receipt.get("adapter", {}).get("mode") != "frozen_base_no_adapter"
    or adapted_receipt.get("adapter", {}).get("enabled") is not True
    or adapted_receipt.get("adapter", {}).get("mode") != "lora_safe_merge"
    or adapted_receipt.get("adapter", {}).get("adapter_model_sha256") != expected_adapter_sha
    or adapted_receipt.get("adapter", {}).get("training_global_step") != 644
    or adapted_receipt.get("adapter", {}).get("lora_rank") != 64
    or adapted_receipt.get("adapter", {}).get("lora_alpha") != 64
):
    raise RuntimeError("base/full644 adapter distinction differs")
for field in ("input", "preprocessing", "prompt_contract", "sampling"):
    if base_receipt.get(field) != adapted_receipt.get(field):
        raise RuntimeError(f"matched pair differs on {field}")

probes = {}
for name in ("case00-source.mp4", "case00-base.mp4", "case00-full644.mp4"):
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,codec_type,width,height,pix_fmt,avg_frame_rate,r_frame_rate,nb_read_frames:format=duration,size",
        "-of",
        "json",
        str(raw_root / name),
    ]
    completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    probe = json.loads(completed.stdout.decode("utf-8", "strict"))
    streams = probe.get("streams")
    if type(streams) is not list or len(streams) != 1 or type(streams[0]) is not dict:
        raise RuntimeError(f"not exactly one media stream: {name}")
    stream = streams[0]
    if (
        stream.get("codec_type") != "video"
        or stream.get("nb_read_frames") != "81"
        or stream.get("avg_frame_rate") != "25/1"
        or type(stream.get("width")) is not int
        or stream["width"] <= 0
        or type(stream.get("height")) is not int
        or stream["height"] <= 0
    ):
        raise RuntimeError(f"not exact 81-frame 25-FPS video: {name}")
    probes[name] = probe
    (stage / "probe" / (name + ".ffprobe.json")).write_text(
        json.dumps(probe, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

base_stream = probes["case00-base.mp4"]["streams"][0]
adapted_stream = probes["case00-full644.mp4"]["streams"][0]
if (base_stream["width"], base_stream["height"]) != (adapted_stream["width"], adapted_stream["height"]):
    raise RuntimeError("matched pair output geometry differs")

manifest = {
    "schema_version": "full644-node233-case00-local-visual-qa-stage-v1",
    "status": "DECODE_VALIDATED_AWAITING_MANUAL_FULL_81_FRAME_REVIEW",
    "remote_root": remote_root,
    "source": {
        "path": source_path,
        "sha256": expected_source_sha,
        "iid": "1852ada01d7c43a4",
    },
    "instruction": "Show the car driving dynamically through the snowy landscape, kicking up snow.",
    "instruction_sha256": expected_instruction_sha,
    "seed": 2026,
    "controller_evidence_digest": evidence_digest,
    "report_digest": report_digest,
    "attestation_digest": attestation_digest,
    "videos": {
        name: {
            "sha256": sha256(raw_root / name),
            "size": (raw_root / name).stat().st_size,
            "ffprobe": probes[name],
        }
        for name in ("case00-source.mp4", "case00-base.mp4", "case00-full644.mp4")
    },
    "all_81_frames_extracted": False,
    "manual_visual_review_complete": False,
    "html_generated": False,
}
(stage / "qa_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY_VALIDATE

for stem in source base full644; do
  input="$stage/raw/case00-$stem.mp4"
  frame_dir="$stage/frames/$stem"
  mkdir -p "$frame_dir"
  ffmpeg -hide_banner -loglevel error -nostdin -i "$input" -map 0:v:0 -fps_mode passthrough -start_number 0 "$frame_dir/frame_%03d.png"
  count="$(find "$frame_dir" -mindepth 1 -maxdepth 1 -type f -name 'frame_*.png' | wc -l | tr -d ' ')"
  [[ "$count" == "81" ]] || {
    printf 'decoded frame count differs for %s: %s\n' "$stem" "$count" >&2
    exit 20
  }
  for index in $(seq 0 80); do
    printf -v leaf 'frame_%03d.png' "$index"
    [[ -s "$frame_dir/$leaf" ]] || {
      printf 'missing decoded frame for %s: %s\n' "$stem" "$leaf" >&2
      exit 21
    }
  done
  ffmpeg -hide_banner -loglevel error -nostdin -y -i "$input" \
    -vf "select='eq(n,0)+eq(n,20)+eq(n,40)+eq(n,60)+eq(n,80)',scale=320:-2,tile=5x1:padding=2:margin=2" \
    -frames:v 1 "$stage/contact_sheets/key5/$stem-frames-000-020-040-060-080.png"
  ffmpeg -hide_banner -loglevel error -nostdin -y -i "$input" \
    -vf "scale=160:-2,tile=9x9:padding=2:margin=2" \
    -frames:v 1 "$stage/contact_sheets/all81/$stem-all-81-frames.png"
done

/usr/bin/python3 - "$stage/qa_manifest.json" <<'PY_FINALIZE'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["all_81_frames_extracted"] = True
value["contact_sheets_generated"] = {
    "key5_indices": [0, 20, 40, 60, 80],
    "all81_layout": "9x9",
}
path.write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY_FINALIZE

printf 'QA_STAGE=%s\n' "$stage"
printf 'STATUS=DECODE_VALIDATED_AWAITING_MANUAL_FULL_81_FRAME_REVIEW\n'
printf 'KEY5=%s\n' "$stage/contact_sheets/key5"
printf 'ALL81=%s\n' "$stage/contact_sheets/all81"
printf 'FRAMES=%s\n' "$stage/frames"
printf 'HTML_GENERATED=false\n'

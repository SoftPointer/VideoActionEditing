#!/usr/bin/env bash
# Materialize browser-compatible H.264 proxies from a sealed review source packet.

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 /absolute/review-packet-root" >&2
  exit 64
fi

packet_root="$1"
plan="$packet_root/transcode-plan.json"
if [[ "$packet_root" != /* || ! -d "$packet_root" || -L "$packet_root" ]]; then
  echo "review packet must be an absolute plain directory" >&2
  exit 65
fi
if [[ ! -f "$plan" || -L "$plan" ]]; then
  echo "transcode plan is absent or linked: $plan" >&2
  exit 66
fi
for command in jq ffmpeg ffprobe; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 69
  fi
done

worker() {
  local root="$1"
  local source_relative="$2"
  local destination_relative="$3"
  local transform="$4"
  local source="$root/$source_relative"
  local destination="$root/$destination_relative"
  local temporary="${destination%.mp4}.tmp.mp4"
  if [[ "$source_relative" == /* || "$destination_relative" == /* \
      || "$source_relative" == *".."* || "$destination_relative" == *".."* \
      || "$destination_relative" != *.mp4 ]]; then
    echo "unsafe transcode row: $source_relative -> $destination_relative" >&2
    return 65
  fi
  if [[ ! -f "$source" || -L "$source" || -e "$destination" || -e "$temporary" ]]; then
    echo "source/output state is invalid: $source -> $destination" >&2
    return 66
  fi
  mkdir -p "${destination%/*}"
  local filter="scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2"
  if [[ "$transform" == "reverse" ]]; then
    filter="reverse,$filter"
  elif [[ "$transform" != "normal" ]]; then
    echo "unknown transform: $transform" >&2
    return 65
  fi
  ffmpeg -hide_banner -loglevel error -nostdin -y \
    -i "$source" -map 0:v:0 -an -vf "$filter" \
    -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p \
    -movflags +faststart "$temporary"
  mv "$temporary" "$destination"
}
export -f worker

jq -e '
  type == "array" and length > 0 and
  all(.[];
    (.source | type == "string") and
    (.destination | type == "string") and
    (.transform == "normal" or .transform == "reverse")
  )
' "$plan" >/dev/null

jq -r '.[] | [.source, .destination, .transform] | @tsv' "$plan" \
  | xargs -P 8 -n 3 bash -c 'worker "$0" "$1" "$2" "$3"' "$packet_root"

expected="$(jq 'length' "$plan")"
actual="$(find "$packet_root/media" -type f -name '*.mp4' | wc -l | tr -d ' ')"
if [[ "$actual" != "$expected" ]]; then
  echo "proxy count differs: expected=$expected actual=$actual" >&2
  exit 67
fi

while IFS= read -r media; do
  codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$media")"
  audio_count="$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$media" | wc -l | tr -d ' ')"
  if [[ "$codec" != "h264" || "$audio_count" != "0" ]]; then
    echo "proxy codec contract differs: $media codec=$codec audio=$audio_count" >&2
    exit 68
  fi
done < <(find "$packet_root/media" -type f -name '*.mp4' | sort)

echo "materialized $actual H.264 review proxies"

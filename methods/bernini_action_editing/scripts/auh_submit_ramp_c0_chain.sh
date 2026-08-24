#!/usr/bin/env bash
set -Eeuo pipefail

launcher="${BERNINI_RAMP_C0_LAUNCHER:?set absolute BERNINI_RAMP_C0_LAUNCHER}"
canary_output="${BERNINI_RAMP_C0_CANARY_OUTPUT:?set fresh absolute canary output}"
c0_output="${BERNINI_RAMP_C0_AFTEROK_OUTPUT:?set fresh absolute C0 output}"

[[ "${launcher}" == /* && -f "${launcher}" && ! -L "${launcher}" ]] || {
  echo "invalid RAMP C0 launcher" >&2
  exit 2
}
for output in "${canary_output}" "${c0_output}"; do
  [[ "${output}" == /* && ! -e "${output}" && ! -L "${output}" ]] || {
    echo "RAMP C0 outputs must be fresh absolute paths" >&2
    exit 2
  }
done
[[ "${canary_output}" != "${c0_output}" ]] || {
  echo "canary and afterok outputs must differ" >&2
  exit 2
}

canary_job="$(sbatch --parsable \
  --export="ALL,BERNINI_RAMP_C0_MODE=engineering-canary,BERNINI_RAMP_C0_OUTPUT=${canary_output}" \
  "${launcher}")"
[[ "${canary_job}" =~ ^[0-9]+$ ]] || {
  echo "invalid canary job id: ${canary_job}" >&2
  exit 2
}

c0_job="$(sbatch --parsable \
  --dependency="afterok:${canary_job}" \
  --export="ALL,BERNINI_RAMP_C0_MODE=afterok-c0,BERNINI_RAMP_C0_OUTPUT=${c0_output}" \
  "${launcher}")"
[[ "${c0_job}" =~ ^[0-9]+$ ]] || {
  echo "invalid afterok C0 job id: ${c0_job}" >&2
  exit 2
}

printf 'canary_job=%s\nafterok_c0_job=%s\ncanary_output=%s\nafterok_c0_output=%s\n' \
  "${canary_job}" "${c0_job}" "${canary_output}" "${c0_output}"

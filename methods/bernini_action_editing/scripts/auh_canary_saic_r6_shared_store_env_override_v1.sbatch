#!/usr/bin/env bash
#SBATCH --job-name=saic-r6-env-cny
#SBATCH --partition=faculty
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=00:05:00

set -Eeuo pipefail

formal_launcher="${SAIC_R6_ENV_CANARY_FORMAL_LAUNCHER:?}"
formal_launcher_sha256="${SAIC_R6_ENV_CANARY_FORMAL_LAUNCHER_SHA256:?}"
python_bin="${SAIC_R6_ENV_CANARY_PYTHON:?}"
output_parent="${SAIC_R6_ENV_CANARY_OUTPUT_PARENT:?}"
inherited_value="${TORCH_DISABLE_SHARE_RDZV_TCP_STORE-<absent>}"

fail() {
  echo "[saic-r6-env-canary] ERROR: $*" >&2
  exit 2
}

for name in formal_launcher python_bin output_parent; do
  [[ "${!name}" == /* ]] || fail "${name} must be absolute"
done
[[ "${formal_launcher_sha256}" =~ ^[0-9a-f]{64}$ ]] || fail "formal launcher SHA-256 differs"
[[ "${SLURM_JOB_ID:?launcher requires Slurm}" =~ ^[0-9]+$ ]] || fail "Slurm job ID differs"
[[ "${inherited_value}" == 1 ]] || fail "job entry did not inherit explicit poison value 1"

formal_launcher="$(realpath -e -- "${formal_launcher}")"
python_bin="$(realpath -e -- "${python_bin}")"
output_parent="$(realpath -e -- "${output_parent}")"
[[ -f "${formal_launcher}" && ! -L "${formal_launcher}" ]] || fail "formal launcher differs"
[[ "$(stat -Lc '%a:%h' -- "${formal_launcher}")" == 444:1 ]] || fail "formal launcher mode/link differs"
[[ "$(sha256sum "${formal_launcher}" | awk '{print $1}')" == "${formal_launcher_sha256}" ]] || fail "formal launcher bytes differ"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python differs"
[[ -d "${output_parent}" && ! -L "${output_parent}" ]] || fail "output parent differs"

output_root="${output_parent}/job-${SLURM_JOB_ID}"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "canary output must be fresh"
mkdir -m 0700 -- "${output_root}"

# Bind the override proof to the exact frozen formal launcher, not merely to a
# second independently chosen value in this canary.
"${python_bin}" -B - "${formal_launcher}" <<'PY'
from pathlib import Path
import sys

payload = Path(sys.argv[1]).read_bytes()
needle = b"export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0\n"
if payload.count(needle) != 1:
    raise SystemExit("formal launcher shared-TCPStore assignment differs")
PY

export TORCH_DISABLE_SHARE_RDZV_TCP_STORE=0
observed_child="$(${python_bin} -B - <<'PY'
import os
print(os.environ.get("TORCH_DISABLE_SHARE_RDZV_TCP_STORE", "<absent>"))
PY
)"
[[ "${observed_child}" == 0 ]] || fail "child process did not observe override value 0"

receipt="${output_root}/shared-store-env-override-receipt.json"
"${python_bin}" -B - "${receipt}" "${SLURM_JOB_ID}" "${SLURMD_NODENAME:-$(hostname)}" \
  "${formal_launcher}" "${formal_launcher_sha256}" "${inherited_value}" "${observed_child}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

receipt = Path(sys.argv[1])
core = {
    "schema_version": "saic-r6-shared-store-env-override-canary-v1",
    "status": "explicit_slurm_poison_observed_then_frozen_launcher_override_observed",
    "slurm_job_id": sys.argv[2],
    "node": sys.argv[3],
    "formal_launcher": sys.argv[4],
    "formal_launcher_sha256": sys.argv[5],
    "job_entry_inherited_torch_disable_share_rdzv_tcp_store": sys.argv[6],
    "post_override_child_observed_torch_disable_share_rdzv_tcp_store": sys.argv[7],
    "sbatch_export_replaced_environment": True,
    "poison_was_explicitly_exported_by_submitter": True,
    "formal_launcher_assignment_exactly_present_once": True,
    "shared_tcp_store_enabled_after_override": True,
    "scientific_runtime_entered": False,
    "scientific_output_created": False,
    "authority": {
        "scientific": False,
        "generation": False,
        "training": False,
        "publication": False,
    },
}
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("ascii")
value = {**core, "receipt_digest": hashlib.sha256(canonical(core)).hexdigest()}
payload = canonical(value) + b"\n"
fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
offset = 0
while offset < len(payload):
    wrote = os.write(fd, payload[offset:])
    if wrote <= 0:
        raise SystemExit("receipt write stalled")
    offset += wrote
os.fsync(fd)
os.fchmod(fd, 0o444)
os.fsync(fd)
info = os.fstat(fd)
os.close(fd)
if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
    raise SystemExit("terminal receipt identity differs")
print(receipt)
PY

echo "SAIC_R6_SHARED_STORE_ENV_OVERRIDE_CANARY_OK output=${output_root}"

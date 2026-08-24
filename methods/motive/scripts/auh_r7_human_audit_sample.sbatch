#!/usr/bin/env bash
#SBATCH --job-name=m7-haudit-sample
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -Eeuo pipefail

source_snapshot="${MOTIVE_SOURCE_SNAPSHOT:?set MOTIVE_SOURCE_SNAPSHOT}"
source_tree_sha256="${MOTIVE_SOURCE_TREE_SHA256:?set MOTIVE_SOURCE_TREE_SHA256}"
source_dir="${MOTIVE_R7_HUMAN_AUDIT_SOURCE:?set MOTIVE_R7_HUMAN_AUDIT_SOURCE}"
source_artifact_digest="${MOTIVE_R7_HUMAN_AUDIT_SOURCE_DIGEST:?set MOTIVE_R7_HUMAN_AUDIT_SOURCE_DIGEST}"
source_input_sha256="${MOTIVE_R7_HUMAN_AUDIT_SOURCE_INPUT_SHA256:?set MOTIVE_R7_HUMAN_AUDIT_SOURCE_INPUT_SHA256}"
data_root="${MOTIVE_DATA_ROOT:?set MOTIVE_DATA_ROOT}"
output_dir="${MOTIVE_R7_HUMAN_AUDIT_OUTPUT:?set MOTIVE_R7_HUMAN_AUDIT_OUTPUT}"
primary_reviewer_id="${MOTIVE_R7_PRIMARY_REVIEWER_ID:?set MOTIVE_R7_PRIMARY_REVIEWER_ID}"
secondary_reviewer_id="${MOTIVE_R7_SECONDARY_REVIEWER_ID:?set MOTIVE_R7_SECONDARY_REVIEWER_ID}"
implementation_bundle_digest="${MOTIVE_R7_HUMAN_AUDIT_IMPLEMENTATION_BUNDLE_DIGEST:?set MOTIVE_R7_HUMAN_AUDIT_IMPLEMENTATION_BUNDLE_DIGEST}"
preregistration="${MOTIVE_R7_HUMAN_AUDIT_PREREGISTRATION:?set MOTIVE_R7_HUMAN_AUDIT_PREREGISTRATION}"
preregistration_sha256="${MOTIVE_R7_HUMAN_AUDIT_PREREGISTRATION_SHA256:?set MOTIVE_R7_HUMAN_AUDIT_PREREGISTRATION_SHA256}"
receipt="${MOTIVE_R7_HUMAN_AUDIT_RECEIPT:?set MOTIVE_R7_HUMAN_AUDIT_RECEIPT}"
python_bin="${PYTHON_BIN:?set PYTHON_BIN}"

fail() {
  echo "[r7-human-audit-sample] $*" >&2
  exit 2
}

for digest in \
  "${source_tree_sha256}" \
  "${source_artifact_digest}" \
  "${source_input_sha256}" \
  "${implementation_bundle_digest}" \
  "${preregistration_sha256}"; do
  if [[ ! "${digest}" =~ ^[0-9a-f]{64}$ ]]; then
    fail "all external digests must be lowercase SHA-256"
  fi
done
for absolute_path in \
  "${source_snapshot}" \
  "${source_dir}" \
  "${data_root}" \
  "${output_dir}" \
  "${preregistration}" \
  "${receipt}" \
  "${python_bin}"; do
  if [[ "${absolute_path}" != /* ]]; then
    fail "all paths must be absolute: ${absolute_path}"
  fi
done

for required in \
  "${source_snapshot}/SOURCE_FILES.jsonl" \
  "${source_snapshot}/methods/motive/motive/r7_human_audit_policy.py" \
  "${source_snapshot}/methods/motive/motive/r7_human_audit_sample.py" \
  "${source_snapshot}/methods/motive/motive/r7_human_audit_report.py" \
  "${source_snapshot}/methods/motive/motive/human_review.py" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  "${source_dir}/positives.jsonl" \
  "${source_dir}/negatives.jsonl" \
  "${source_dir}/review.jsonl" \
  "${source_dir}/summary.json" \
  "${source_dir}/done.json" \
  "${data_root}" \
  "${preregistration}"; do
  if [[ ! -e "${required}" ]]; then
    fail "missing required path: ${required}"
  fi
done
if [[ ! -x "${python_bin}" ]]; then
  fail "PYTHON_BIN is not executable: ${python_bin}"
fi
if [[ -L "${preregistration}" || ! -f "${preregistration}" ]]; then
  fail "preregistration must be a regular non-symlink file"
fi
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  fail "refusing existing sample output: ${output_dir}"
fi
if [[ -e "${receipt}" || -L "${receipt}" ]]; then
  fail "refusing existing external sample receipt: ${receipt}"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${source_snapshot}/methods/motive"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

"${python_bin}" - \
  "${preregistration}" \
  "${preregistration_sha256}" \
  "${source_snapshot}" \
  "${source_tree_sha256}" \
  "${source_dir}" \
  "${source_artifact_digest}" \
  "${source_input_sha256}" \
  "${data_root}" \
  "${output_dir}" \
  "${primary_reviewer_id}" \
  "${secondary_reviewer_id}" \
  "${implementation_bundle_digest}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

from motive.human_review import normalize_reviewer_id
from motive.r7_human_audit_policy import (
    implementation_bundle_sha256,
    policy_payload,
    policy_sha256,
)

(
    preregistration_raw,
    expected_preregistration_sha,
    snapshot_raw,
    expected_tree,
    source_raw,
    expected_source_artifact,
    expected_source_input,
    data_raw,
    output_raw,
    primary_raw,
    secondary_raw,
    expected_bundle,
) = sys.argv[1:]
preregistration = pathlib.Path(preregistration_raw).resolve(strict=True)
digest = hashlib.sha256(preregistration.read_bytes()).hexdigest()
if digest != expected_preregistration_sha:
    raise SystemExit(
        "[r7-human-audit-sample] preregistration SHA differs"
    )
payload = json.loads(preregistration.read_text(encoding="utf-8"))
expected_fields = {
    "schema_version",
    "created_at_utc",
    "source_snapshot",
    "source_tree_sha256",
    "implementation_bundle_sha256",
    "policy_sha256",
    "source_directory",
    "source_artifact_digest",
    "source_input_sha256",
    "data_root",
    "output_directory",
    "primary_reviewer_id",
    "secondary_reviewer_id",
    "sampling_design",
    "training_authorized",
}
if set(payload) != expected_fields:
    raise SystemExit(
        "[r7-human-audit-sample] preregistration fields differ"
    )
primary = normalize_reviewer_id(primary_raw, context="primary")
secondary = normalize_reviewer_id(secondary_raw, context="secondary")
policy = policy_payload()
expected = {
    "schema_version": "motive-r7-human-audit-preregistration-v1",
    "source_snapshot": str(
        pathlib.Path(snapshot_raw).resolve(strict=True)
    ),
    "source_tree_sha256": expected_tree,
    "implementation_bundle_sha256": expected_bundle,
    "policy_sha256": policy_sha256(),
    "source_directory": str(
        pathlib.Path(source_raw).resolve(strict=True)
    ),
    "source_artifact_digest": expected_source_artifact,
    "source_input_sha256": expected_source_input,
    "data_root": str(pathlib.Path(data_raw).resolve(strict=True)),
    "output_directory": str(
        pathlib.Path(output_raw).resolve(strict=False)
    ),
    "primary_reviewer_id": primary,
    "secondary_reviewer_id": secondary,
    "sampling_design": policy["sampling_design"],
    "training_authorized": False,
}
if not isinstance(payload.get("created_at_utc"), str) or not re.fullmatch(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)",
    payload["created_at_utc"],
):
    raise SystemExit(
        "[r7-human-audit-sample] preregistration timestamp differs"
    )
for key, value in expected.items():
    if payload.get(key) != value:
        raise SystemExit(
            f"[r7-human-audit-sample] preregistration {key} differs"
        )
if implementation_bundle_sha256() != expected_bundle:
    raise SystemExit(
        "[r7-human-audit-sample] implementation bundle differs"
    )
print(
    "[r7-human-audit-sample] preregistration verified "
    f"sha256={digest}",
    flush=True,
)
PY

echo "[r7-human-audit-sample] host=$(hostname) job=${SLURM_JOB_ID}"
echo "[r7-human-audit-sample] output=${output_dir}"

"${python_bin}" -m motive.r7_human_audit_sample \
  --source-dir "${source_dir}" \
  --output-dir "${output_dir}" \
  --data-root "${data_root}" \
  --primary-reviewer-id "${primary_reviewer_id}" \
  --secondary-reviewer-id "${secondary_reviewer_id}" \
  --expected-implementation-bundle-digest \
    "${implementation_bundle_digest}" \
  --expected-source-artifact-digest "${source_artifact_digest}" \
  --expected-source-input-sha256 "${source_input_sha256}"

# Verification-only replay rederives the frozen sample and rehashes live media.
"${python_bin}" -m motive.r7_human_audit_sample \
  --source-dir "${source_dir}" \
  --output-dir "${output_dir}" \
  --data-root "${data_root}" \
  --primary-reviewer-id "${primary_reviewer_id}" \
  --secondary-reviewer-id "${secondary_reviewer_id}" \
  --expected-implementation-bundle-digest \
    "${implementation_bundle_digest}" \
  --expected-source-artifact-digest "${source_artifact_digest}" \
  --expected-source-input-sha256 "${source_input_sha256}" \
  --resume

# A second independent replay is the terminal sample check.  It must remain
# before source verification and receipt publication so no successful receipt
# can describe an output that was not replayed twice from the frozen inputs.
"${python_bin}" -m motive.r7_human_audit_sample \
  --source-dir "${source_dir}" \
  --output-dir "${output_dir}" \
  --data-root "${data_root}" \
  --primary-reviewer-id "${primary_reviewer_id}" \
  --secondary-reviewer-id "${secondary_reviewer_id}" \
  --expected-implementation-bundle-digest \
    "${implementation_bundle_digest}" \
  --expected-source-artifact-digest "${source_artifact_digest}" \
  --expected-source-input-sha256 "${source_input_sha256}" \
  --resume

"${python_bin}" \
  "${source_snapshot}/methods/motive/scripts/action_source_snapshot.py" \
  verify \
  --snapshot "${source_snapshot}" \
  --expected-tree-sha256 "${source_tree_sha256}"

sha256sum \
  "${preregistration}" \
  "${output_dir}/sampled_manifest.jsonl" \
  "${output_dir}/secondary_manifest.jsonl" \
  "${output_dir}/primary_review.blind.jsonl" \
  "${output_dir}/secondary_review.blind.jsonl" \
  "${output_dir}/summary.json" \
  "${output_dir}/done.json"

# The external receipt is deliberately the final state transition.  Every
# replay, live-media rehash, source-snapshot verification, and artifact read
# above must succeed before O_EXCL publishes it.
mkdir -p "$(dirname "${receipt}")"
"${python_bin}" - \
  "${output_dir}" \
  "${receipt}" \
  "${preregistration}" \
  "${preregistration_sha256}" \
  "${source_snapshot}" \
  "${source_tree_sha256}" \
  "${source_artifact_digest}" \
  "${source_input_sha256}" \
  "${implementation_bundle_digest}" \
  "${primary_reviewer_id}" \
  "${secondary_reviewer_id}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

(
    output_raw,
    receipt_raw,
    preregistration_raw,
    preregistration_sha,
    snapshot_raw,
    source_tree_sha,
    source_artifact_digest,
    source_input_sha,
    implementation_bundle_sha,
    primary_reviewer,
    secondary_reviewer,
) = sys.argv[1:]
output = pathlib.Path(output_raw).resolve(strict=True)
receipt = pathlib.Path(receipt_raw).resolve(strict=False)
if receipt == output or output in receipt.parents:
    raise SystemExit(
        "[r7-human-audit-sample] receipt must be external to sample"
    )
done_path = output / "done.json"
summary_path = output / "summary.json"
done = json.loads(done_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
design = done.get("sampling_design", {})
selected = summary.get("selected", {})
if (
    done.get("status") != "complete"
    or done.get("media_bytes_bound") is not True
    or done.get("implementation_bundle_external_anchor_verified")
    is not True
    or done.get("source_external_anchor_verified") is not True
    or done.get("formal_gate_input_eligible") is not True
    or design.get("mode") != "formal_policy_locked"
    or design.get("matches_immutable_policy") is not True
    or selected.get("pseudo_positive_probability") != 240
    or selected.get("pseudo_negative_probability") != 200
    or selected.get("priority_review") != 80
    or done.get("training_authorized") is not False
):
    raise SystemExit(
        "[r7-human-audit-sample] formal sample contract differs"
    )
def sha(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
payload = {
    "schema_version": "motive-r7-human-audit-sample-receipt-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "source_snapshot": str(
        pathlib.Path(snapshot_raw).resolve(strict=True)
    ),
    "source_tree_sha256": source_tree_sha,
    "preregistration": str(
        pathlib.Path(preregistration_raw).resolve(strict=True)
    ),
    "preregistration_sha256": preregistration_sha,
    "sample_directory": str(output),
    "sample_artifact_digest": done["artifact_digest"],
    "sample_done_sha256": sha(done_path),
    "sample_summary_sha256": sha(summary_path),
    "source_artifact_digest": source_artifact_digest,
    "source_input_sha256": source_input_sha,
    "implementation_bundle_sha256": implementation_bundle_sha,
    "policy_sha256": done["policy_sha256"],
    "assignment_set_digest": done["assignment_set_digest"],
    "media_binding_set_digest": done["media_binding_set_digest"],
    "primary_reviewer_id": done["primary_reviewer_id"],
    "secondary_reviewer_id": done["secondary_reviewer_id"],
    "sample_rows": done["rows"],
    "secondary_rows": selected["double_review"],
    "formal_gate_input_eligible": True,
    "training_authorized": False,
    "threat_model": done["threat_model"],
}
receipt.parent.mkdir(parents=True, exist_ok=True)
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
descriptor = os.open(receipt, flags, 0o444)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
finally:
    if not receipt.exists():
        try:
            os.close(descriptor)
        except OSError:
            pass
print(
    "[r7-human-audit-sample] completed; external receipt "
    f"artifact_digest={done['artifact_digest']} path={receipt}",
    flush=True,
)
PY

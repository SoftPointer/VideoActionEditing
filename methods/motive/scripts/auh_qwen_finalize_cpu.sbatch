#!/usr/bin/env bash
#SBATCH --job-name=motive-qwen-merge
#SBATCH --partition=faculty
#SBATCH --account=test-acc
#SBATCH --qos=bgqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# AUH faculty QOS enforces MinTRES=gres/gpu:1 for every job.
#SBATCH --gres=gpu:mi210:1
#SBATCH --time=04:00:00

set -Eeuo pipefail

repo_root="${MOTIVE_REPO_ROOT:-/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/VideoEdit}"
output_root="${MOTIVE_OUTPUT_ROOT:-${repo_root}/methods/motive/outputs/goku_subject_movement_full}"
code_root="${MOTIVE_CODE_ROOT:-${repo_root}/methods/motive}"
python_bin="${PYTHON_BIN:-/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python}"
shard_count="${MOTIVE_QWEN_NUM_SHARDS:-8}"
qwen_input="${MOTIVE_QWEN_INPUT:-${output_root}/feature_candidates/qwen_queue.jsonl}"
feature_dir="${MOTIVE_FUSE_FEATURE_DIR:-${output_root}/features_merged/features}"
qwen_root="${output_root}/qwen_visual"
merged_output="${qwen_root}/qwen_visual.jsonl"
provenance_output="${qwen_root}/qwen_visual.provenance.json"
fused_output="${output_root}/fused"
legacy_provenance="${MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE:-0}"

if [[ "${shard_count}" != "8" ]]; then
  echo "[motive-qwen-merge] expected eight shards, got ${shard_count}" >&2
  exit 2
fi
if [[ "${legacy_provenance}" != "0" && "${legacy_provenance}" != "1" ]]; then
  echo "[motive-qwen-merge] MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE must be 0 or 1" >&2
  exit 2
fi
for required in \
  "${code_root}/motive/cascade.py" \
  "${qwen_input}" \
  "${feature_dir}" \
  "${python_bin}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[motive-qwen-merge] missing required path: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${qwen_root}"
temporary_output="${merged_output}.tmp.${SLURM_JOB_ID:-manual}"
temporary_provenance="${provenance_output}.tmp.${SLURM_JOB_ID:-manual}"
reuse_final=0
if [[ -e "${merged_output}" && -e "${provenance_output}" ]]; then
  reuse_final=1
elif [[ -e "${merged_output}" || -e "${provenance_output}" ]]; then
  echo "[motive-qwen-merge] only one final merge artifact exists; fail closed" >&2
  exit 2
fi
if [[ -e "${temporary_output}" || -e "${temporary_provenance}" ]]; then
  echo "[motive-qwen-merge] stale temporary output exists" >&2
  exit 2
fi
cleanup_temporary_output() {
  if [[ -n "${temporary_output:-}" && -e "${temporary_output}" ]]; then
    rm -f -- "${temporary_output}"
  fi
  if [[ -n "${temporary_provenance:-}" && -e "${temporary_provenance}" ]]; then
    rm -f -- "${temporary_provenance}"
  fi
}
trap cleanup_temporary_output EXIT
if [[ "${reuse_final}" == "1" ]]; then
  validation_input="${merged_output}"
else
  : > "${temporary_output}"
  for ((shard_index = 0; shard_index < shard_count; shard_index++)); do
    shard_tag="$(printf '%03d' "${shard_index}")"
    shard_path="${qwen_root}/shards/qwen-${shard_tag}.jsonl"
    if [[ ! -s "${shard_path}" ]]; then
      echo "[motive-qwen-merge] missing shard: ${shard_path}" >&2
      exit 2
    fi
    cat "${shard_path}" >> "${temporary_output}"
  done
  validation_input="${temporary_output}"
fi

"${python_bin}" - \
  "${qwen_input}" \
  "${validation_input}" \
  "${MOTIVE_QWEN_ALLOW_ERRORS:-0}" \
  "${code_root}" \
  "${temporary_provenance}" \
  "${qwen_root}" \
  "${shard_count}" \
  "${legacy_provenance}" \
  "${reuse_final}" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

code_root = Path(sys.argv[4]).resolve()
sys.path.insert(0, str(code_root))
from motive.qwen_filter import (  # noqa: E402
    OBSERVATION_SCHEMA_VERSION,
    VISUAL_SCHEMA_VERSION,
    _object_digest,
    _validate_observation,
    _validate_visual,
)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    path = Path(path)
    raw = path.read_bytes()
    physical_lines = raw.splitlines(keepends=True)
    rows = []
    for line_number, raw_line in enumerate(physical_lines, start=1):
        if not raw_line.strip():
            raise RuntimeError(f"{path}:{line_number} is blank")
        try:
            line = raw_line.decode("utf-8")
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{path}:{line_number} is not valid UTF-8 JSON"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows, raw, physical_lines


def row_iid(row, *, source):
    raw_iid = row.get("iid")
    if raw_iid is None and source:
        raw_iid = row.get("id")
    if raw_iid is None:
        raise RuntimeError("JSONL row is missing iid/id")
    return str(raw_iid)


def require_unique_iids(rows, *, source, description):
    values = [row_iid(row, source=source) for row in rows]
    if len(values) != len(set(values)):
        duplicates = sorted(
            iid for iid, count in Counter(values).items() if count > 1
        )[:10]
        raise RuntimeError(
            f"{description} contains duplicate iid values: {duplicates}"
        )
    return values


input_path = Path(sys.argv[1])
merged_path = Path(sys.argv[2])
allow_errors = sys.argv[3] == "1"
provenance_path = Path(sys.argv[5])
qwen_root = Path(sys.argv[6])
shard_count = int(sys.argv[7])
allow_legacy = sys.argv[8] == "1"
reuse_final = sys.argv[9] == "1"

input_rows, input_raw, input_lines = load(input_path)
result_rows, merged_raw, _ = load(merged_path)
if not input_rows or not result_rows:
    raise RuntimeError("Qwen merge refuses an empty input or result set")
input_ids = require_unique_iids(
    input_rows,
    source=True,
    description="Qwen input",
)
result_ids = require_unique_iids(
    result_rows,
    source=False,
    description="merged Qwen output",
)
input_digests = {}
for row, iid in zip(input_rows, input_ids):
    input_digest = row.get("input_digest")
    if not isinstance(input_digest, str) or not input_digest:
        raise RuntimeError(f"Qwen input lacks input_digest for iid={iid}")
    input_digests[iid] = input_digest
if set(input_ids) != set(result_ids):
    missing = sorted(set(input_ids) - set(result_ids))[:10]
    extra = sorted(set(result_ids) - set(input_ids))[:10]
    raise RuntimeError(f"Qwen coverage mismatch: missing={missing} extra={extra}")
for row in result_rows:
    iid = str(row["iid"])
    if row.get("input_digest") != input_digests[iid]:
        raise RuntimeError(f"Qwen input digest mismatch for iid={iid}")

source_sha256 = sha256_bytes(input_raw)
source_rows = len(input_rows)
legacy_reasons = set()
shard_audits = []
all_shard_rows = []
all_shard_raw = bytearray()
expected_all_ids = set()

for shard_index in range(shard_count):
    shard_tag = f"{shard_index:03d}"
    manifest_path = qwen_root / "manifests" / f"shard-{shard_tag}.jsonl"
    marker_path = Path(f"{manifest_path}.source")
    output_path = qwen_root / "shards" / f"qwen-{shard_tag}.jsonl"
    for required_path in (manifest_path, marker_path, output_path):
        if not required_path.is_file():
            raise RuntimeError(f"missing shard artifact: {required_path}")

    manifest_rows, manifest_raw, _ = load(manifest_path)
    output_rows, output_raw, _ = load(output_path)
    expected_lines = [
        line if line.endswith(b"\n") else line + b"\n"
        for line in input_lines[shard_index::shard_count]
    ]
    expected_manifest_raw = b"".join(expected_lines)
    if manifest_raw != expected_manifest_raw:
        raise RuntimeError(
            f"shard manifest bytes do not match line-modulo source partition: "
            f"shard={shard_index}"
        )

    manifest_sha256 = sha256_bytes(manifest_raw)
    marker_raw = marker_path.read_bytes()
    marker_sha256 = sha256_bytes(marker_raw)
    output_sha256 = sha256_bytes(output_raw)
    marker_text = marker_raw.decode("utf-8").strip()
    expected_marker = {
        "partition": "line_modulo_v1",
        "schema_version": "motive-qwen-shard-manifest-v2",
        "shard_count": shard_count,
        "shard_index": shard_index,
        "shard_rows": len(manifest_rows),
        "shard_sha256": manifest_sha256,
        "source_rows": source_rows,
        "source_sha256": source_sha256,
    }
    try:
        marker = json.loads(marker_text)
    except json.JSONDecodeError:
        marker = None
    if isinstance(marker, dict):
        if marker != expected_marker:
            raise RuntimeError(
                f"shard marker mismatch for shard={shard_index}: "
                f"expected={expected_marker!r} actual={marker!r}"
            )
        marker_version = "motive-qwen-shard-manifest-v2"
        marker_is_legacy = False
    else:
        if not allow_legacy:
            raise RuntimeError(
                f"legacy shard marker refused for shard={shard_index}; set "
                "MOTIVE_QWEN_ALLOW_LEGACY_PROVENANCE=1 only for an audited "
                "in-flight legacy run"
            )
        expected_legacy_marker = (
            f"{source_sha256}|{shard_count}|{shard_index}"
        )
        if marker_text != expected_legacy_marker:
            raise RuntimeError(
                f"legacy shard marker mismatch for shard={shard_index}"
            )
        marker_version = "legacy-source-marker-v1"
        marker_is_legacy = True
        legacy_reasons.add("legacy_shard_marker_v1")

    manifest_ids = require_unique_iids(
        manifest_rows,
        source=True,
        description=f"shard {shard_index} manifest",
    )
    output_ids = require_unique_iids(
        output_rows,
        source=False,
        description=f"shard {shard_index} output",
    )
    if set(manifest_ids) != set(output_ids):
        missing = sorted(set(manifest_ids) - set(output_ids))[:10]
        extra = sorted(set(output_ids) - set(manifest_ids))[:10]
        raise RuntimeError(
            f"shard {shard_index} output coverage mismatch: "
            f"missing={missing} extra={extra}"
        )
    if expected_all_ids.intersection(manifest_ids):
        raise RuntimeError(f"iid appears in multiple manifests: shard={shard_index}")
    expected_all_ids.update(manifest_ids)

    manifest_digests = {}
    for row, iid in zip(manifest_rows, manifest_ids):
        manifest_digest = row.get("input_digest")
        if manifest_digest != input_digests.get(iid):
            raise RuntimeError(
                f"shard manifest input digest mismatch: shard={shard_index} "
                f"iid={iid}"
            )
        manifest_digests[iid] = manifest_digest

    missing_execution_rows = 0
    missing_result_digest_rows = 0
    shard_config_digests = {
        str(row.get("config_digest") or "") for row in output_rows
    }
    if len(shard_config_digests) != 1 or not next(iter(shard_config_digests)):
        raise RuntimeError(
            f"Qwen shard has mixed/missing config_digest: shard={shard_index}"
        )
    shard_config_digest = next(iter(shard_config_digests))
    raw_run_config_digests = [
        row.get("run_config_digest") for row in output_rows
    ]
    if all(value is None for value in raw_run_config_digests):
        if not allow_legacy:
            raise RuntimeError(
                f"Qwen shard lacks run_config_digest: shard={shard_index}"
            )
        shard_run_config_digest = None
        legacy_reasons.add("missing_run_config_digest")
    elif any(value is None for value in raw_run_config_digests):
        raise RuntimeError(
            f"Qwen shard mixes legacy/new run_config_digest rows: "
            f"shard={shard_index}"
        )
    else:
        run_config_digests = {
            str(value) for value in raw_run_config_digests
        }
        if len(run_config_digests) != 1 or not next(iter(run_config_digests)):
            raise RuntimeError(
                f"Qwen shard has mixed/empty run_config_digest: "
                f"shard={shard_index}"
            )
        shard_run_config_digest = next(iter(run_config_digests))
    for row in output_rows:
        iid = str(row["iid"])
        if row.get("input_digest") != manifest_digests[iid]:
            raise RuntimeError(
                f"shard output input digest mismatch: shard={shard_index} "
                f"iid={iid}"
            )
        execution_index = row.get("execution_shard_index")
        execution_count = row.get("execution_shard_count")
        execution_manifest_path = row.get("execution_manifest")
        execution_manifest = row.get("execution_manifest_sha256")
        execution_missing = (
            execution_index is None
            and execution_count is None
            and execution_manifest_path is None
            and execution_manifest is None
        )
        if execution_missing:
            if not allow_legacy:
                raise RuntimeError(
                    f"Qwen row lacks execution shard provenance: iid={iid}"
                )
            missing_execution_rows += 1
            legacy_reasons.add("missing_execution_shard_provenance")
        else:
            if (
                type(execution_index) is not int
                or type(execution_count) is not int
                or execution_index != shard_index
                or execution_count != shard_count
                or not isinstance(execution_manifest_path, str)
                or Path(execution_manifest_path).resolve()
                != manifest_path.resolve()
                or execution_manifest != manifest_sha256
            ):
                raise RuntimeError(
                    f"Qwen execution shard provenance mismatch: "
                    f"shard={shard_index} iid={iid}"
                )
        if row.get("status") == "ok" and row.get("result_digest") is None:
            if not allow_legacy:
                raise RuntimeError(f"Qwen ok row lacks result_digest: iid={iid}")
            missing_result_digest_rows += 1
            legacy_reasons.add("missing_result_digest")

    shard_audits.append(
        {
            "index": shard_index,
            "marker": {
                "legacy": marker_is_legacy,
                "rows": len(marker_raw.splitlines()),
                "schema_version": marker_version,
                "sha256": marker_sha256,
            },
            "manifest": {
                "rows": len(manifest_rows),
                "sha256": manifest_sha256,
            },
            "output": {
                "config_digest": shard_config_digest,
                "run_config_digest": shard_run_config_digest,
                "rows": len(output_rows),
                "sha256": output_sha256,
            },
            "legacy_missing_execution_rows": missing_execution_rows,
            "legacy_missing_result_digest_rows": missing_result_digest_rows,
        }
    )
    all_shard_rows.extend(output_rows)
    all_shard_raw.extend(output_raw)

if expected_all_ids != set(input_ids):
    missing = sorted(set(input_ids) - expected_all_ids)[:10]
    extra = sorted(expected_all_ids - set(input_ids))[:10]
    raise RuntimeError(
        f"shard manifests do not partition the Qwen input: "
        f"missing={missing} extra={extra}"
    )

global_run_config_digests = {
    shard["output"]["run_config_digest"]
    for shard in shard_audits
    if shard["output"]["run_config_digest"] is not None
}
missing_run_config_shards = [
    shard["index"]
    for shard in shard_audits
    if shard["output"]["run_config_digest"] is None
]
if missing_run_config_shards and global_run_config_digests:
    raise RuntimeError(
        "Qwen outputs mix legacy/new run_config_digest shards: "
        f"missing={missing_run_config_shards}"
    )
if len(global_run_config_digests) > 1:
    raise RuntimeError(
        "Qwen semantic run configuration differs across shards: "
        f"{sorted(global_run_config_digests)}"
    )
global_run_config_digest = (
    next(iter(global_run_config_digests))
    if global_run_config_digests
    else None
)
legacy_run_config_proxy_digest = None
if global_run_config_digest is None:
    # Before execution-shard provenance was added, config_digest itself was
    # shard-independent. Preserve that available cross-shard consistency
    # check rather than treating compatibility mode as unverified.
    legacy_config_digests = {
        shard["output"]["config_digest"] for shard in shard_audits
    }
    if len(legacy_config_digests) != 1:
        raise RuntimeError(
            "legacy Qwen semantic configuration differs across shards: "
            f"{sorted(legacy_config_digests)}"
        )
    legacy_run_config_proxy_digest = next(iter(legacy_config_digests))

# A reused merge may differ only in ordering/whitespace. Validate semantic row
# identity here, then let the merged byte SHA in provenance make cmp fail.
shard_row_digests = {
    str(row["iid"]): _object_digest(row) for row in all_shard_rows
}
merged_row_digests = {
    str(row["iid"]): _object_digest(row) for row in result_rows
}
if shard_row_digests != merged_row_digests:
    raise RuntimeError("merged Qwen rows differ from the validated shard outputs")
if not reuse_final and merged_raw != bytes(all_shard_raw):
    raise RuntimeError("new merged Qwen bytes differ from shard concatenation")


def unique_field(name):
    values = {str(row.get(name) or "") for row in result_rows}
    if len(values) != 1 or not next(iter(values)):
        raise RuntimeError(f"Qwen rows have mixed/missing {name}: {sorted(values)}")
    return next(iter(values))


contract = {
    "run_config_digest": global_run_config_digest,
    "legacy_run_config_proxy_digest": legacy_run_config_proxy_digest,
    "validator_implementation_sha256": sha256_bytes(
        (code_root / "motive" / "qwen_filter.py").read_bytes()
    ),
    "config_digest_by_shard": {
        str(shard["index"]): shard["output"]["config_digest"]
        for shard in shard_audits
    },
    "implementation_digest": unique_field("implementation_digest"),
    "model_revision": unique_field("model_revision"),
    "transformers_version": unique_field("transformers_version"),
    "mode": unique_field("mode"),
}
if contract["mode"] != "visual":
    raise RuntimeError(f"expected visual Qwen rows, got {contract['mode']!r}")
errors = sum(row.get("status") != "ok" for row in result_rows)
if errors and not allow_errors:
    raise RuntimeError(
        f"merged Qwen output contains {errors} errors; inspect/retry or set "
        "MOTIVE_QWEN_ALLOW_ERRORS=1"
    )
observation_repair_rows = 0
observation_repair_attempts = 0
alignment_repair_rows = 0
alignment_repair_attempts = 0
observation_fallbacks = 0
result_fallbacks = 0
any_fallbacks = 0
result_digest_present = 0
result_digest_verified = 0
strict_semantic_conflict_rows = 0
ok_rows = 0
for row in result_rows:
    observation_repairs = row.get("observation_repairs", [])
    alignment_repairs = row.get("alignment_repairs", [])
    if not isinstance(observation_repairs, list):
        raise RuntimeError(
            f"Qwen observation_repairs is not a list: iid={row.get('iid')}"
        )
    if not isinstance(alignment_repairs, list):
        raise RuntimeError(
            f"Qwen alignment_repairs is not a list: iid={row.get('iid')}"
        )
    observation_repair_rows += bool(observation_repairs)
    observation_repair_attempts += len(observation_repairs)
    alignment_repair_rows += bool(alignment_repairs)
    alignment_repair_attempts += len(alignment_repairs)
    observation_is_fallback = (
        row.get("observation_validated_from") == "fallback_uncertain"
    )
    result_is_fallback = (
        row.get("result_validated_from") == "fallback_uncertain"
    )
    observation_fallbacks += observation_is_fallback
    result_fallbacks += result_is_fallback
    any_fallbacks += observation_is_fallback or result_is_fallback
    if row.get("status") != "ok":
        continue
    ok_rows += 1
    observation = row.get("observation")
    result = row.get("result")
    if not isinstance(observation, dict):
        raise RuntimeError(
            f"Qwen ok row lacks observation object: iid={row.get('iid')}"
        )
    if not isinstance(result, dict):
        raise RuntimeError(f"Qwen ok row lacks result object: iid={row.get('iid')}")
    legacy_schema_row = allow_legacy and (
        row.get("run_config_digest") is None
        or row.get("result_digest") is None
        or row.get("execution_shard_index") is None
        or row.get("execution_shard_count") is None
        or row.get("execution_manifest_sha256") is None
    )
    try:
        _validate_observation(observation)
        if legacy_schema_row:
            _validate_visual(
                result,
                observation=observation,
                require_negative_evidence=False,
            )
        else:
            _validate_visual(result, observation=observation)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Qwen schema validation failed: iid={row.get('iid')}: {error}"
        ) from error
    if legacy_schema_row:
        try:
            _validate_visual(
                result,
                observation=observation,
                require_negative_evidence=True,
            )
        except (TypeError, ValueError):
            # Audit only: retain the immutable legacy row and never promote
            # this compatibility result to strict evidence.
            strict_semantic_conflict_rows += 1
    observation_digest = _object_digest(observation)
    if row.get("observation_digest") != observation_digest:
        raise RuntimeError(
            f"Qwen observation digest mismatch: iid={row.get('iid')}"
        )
    result_digest = row.get("result_digest")
    expected_result_digest = _object_digest(result)
    if result_digest is None:
        if not allow_legacy:
            raise RuntimeError(
                f"Qwen result digest missing: iid={row.get('iid')}"
            )
    else:
        result_digest_present += 1
        if result_digest != expected_result_digest:
            raise RuntimeError(
                f"Qwen result digest mismatch: iid={row.get('iid')}"
            )
        result_digest_verified += 1
    observation_fallback = row.get("observation_fallback")
    if observation_is_fallback:
        if (
            not isinstance(observation_fallback, dict)
            or observation_fallback.get("fallback_digest")
            != observation_digest
        ):
            raise RuntimeError(
                f"Qwen observation fallback audit mismatch: iid={row.get('iid')}"
            )
    for repair in alignment_repairs:
        if repair.get("authoritative_context_digest") != observation_digest:
            raise RuntimeError(
                "Qwen alignment repair used different observation context: "
                f"iid={row.get('iid')}"
            )
    result_fallback = row.get("result_fallback")
    if result_is_fallback:
        if not isinstance(result_fallback, dict):
            raise RuntimeError(
                f"Qwen result fallback audit missing: iid={row.get('iid')}"
            )
        if (
            result_fallback.get("authoritative_context_digest")
            != observation_digest
            or result_fallback.get("fallback_digest")
            != _object_digest(result)
        ):
            raise RuntimeError(
                f"Qwen result fallback audit mismatch: iid={row.get('iid')}"
            )
result_digest_missing = ok_rows - result_digest_present
contract.update(
    {
        "provenance_schema_version": "motive-qwen-merge-provenance-v2",
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "schema_version": VISUAL_SCHEMA_VERSION,
        "allow_errors": allow_errors,
        "input_sha256": source_sha256,
        "input_count": len(input_ids),
        "merged_sha256": sha256_bytes(merged_raw),
        "result_count": len(result_ids),
        "shard_count": shard_count,
        "errors": errors,
        "strict_semantic_conflict_rows": strict_semantic_conflict_rows,
        "repair_counts": {
            "alignment_attempts": alignment_repair_attempts,
            "alignment_rows": alignment_repair_rows,
            "observation_attempts": observation_repair_attempts,
            "observation_rows": observation_repair_rows,
        },
        "fallback_counts": {
            "any_rows": any_fallbacks,
            "observation_rows": observation_fallbacks,
            "result_rows": result_fallbacks,
        },
        "result_digest_coverage": {
            "eligible_ok_rows": ok_rows,
            "missing": result_digest_missing,
            "present": result_digest_present,
            "verified": result_digest_verified,
        },
        "shards": shard_audits,
        "legacy": bool(legacy_reasons),
        "legacy_compatibility": {
            "enabled": allow_legacy,
            "reasons": sorted(legacy_reasons),
            "used": bool(legacy_reasons),
        },
    }
)
provenance_path.write_text(
    json.dumps(contract, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    f"[motive-qwen-merge] inputs={len(input_ids)} results={len(result_ids)} "
    f"errors={errors} fallbacks={any_fallbacks} "
    f"result_digest={result_digest_verified}/{ok_rows} "
    f"strict_conflicts={strict_semantic_conflict_rows} "
    f"legacy={bool(legacy_reasons)}"
)
PY
if [[ "${reuse_final}" == "1" ]]; then
  if ! cmp -s "${temporary_provenance}" "${provenance_output}"; then
    echo "[motive-qwen-merge] existing final provenance no longer validates" >&2
    exit 2
  fi
  rm -f -- "${temporary_provenance}"
  temporary_provenance=""
else
  mv "${temporary_output}" "${merged_output}"
  temporary_output=""
  mv "${temporary_provenance}" "${provenance_output}"
  temporary_provenance=""
fi

cd "${repo_root}"
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
"${python_bin}" -m motive.cascade fuse \
  --feature-dir "${feature_dir}" \
  --output-dir "${fused_output}" \
  --qwen-visual "${merged_output}" \
  --keep-threshold "${MOTIVE_KEEP_THRESHOLD:-0.68}" \
  --review-threshold "${MOTIVE_REVIEW_THRESHOLD:-0.42}"

# Freeze only machine-produced evidence and decisions. Human-review working
# files live outside these paths and deliberately remain writable.
chmod a-w -- "${qwen_input}" "${merged_output}" "${provenance_output}"
find "${qwen_root}/manifests" "${qwen_root}/shards" \
  -type f -exec chmod a-w -- {} +
find "${fused_output}" -type f -exec chmod a-w -- {} +

echo "[motive-qwen-merge] merged=${merged_output}"
echo "[motive-qwen-merge] fused=${fused_output}"
echo "[motive-qwen-merge] automatic artifacts frozen read-only"

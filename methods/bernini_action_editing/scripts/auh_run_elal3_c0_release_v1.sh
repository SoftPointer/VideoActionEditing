#!/usr/bin/env bash
# Execute the three preregistered synthetic ELAL-3 C0 arms from an immutable release.
# This is an engineering canary only; it never performs an optimizer update.

set -Eeuo pipefail
umask 0027

fail() {
  echo "[elal3-c0] ERROR: $*" >&2
  exit 2
}

archive="${ELAL3_C0_SOURCE_ARCHIVE:?set ELAL3_C0_SOURCE_ARCHIVE}"
archive_sha="${ELAL3_C0_SOURCE_ARCHIVE_SHA256:?set ELAL3_C0_SOURCE_ARCHIVE_SHA256}"
manifest="${ELAL3_C0_SOURCE_MANIFEST:?set ELAL3_C0_SOURCE_MANIFEST}"
manifest_sha="${ELAL3_C0_SOURCE_MANIFEST_SHA256:?set ELAL3_C0_SOURCE_MANIFEST_SHA256}"
output_root="${ELAL3_C0_OUTPUT_ROOT:?set fresh ELAL3_C0_OUTPUT_ROOT}"
python_bin="${ELAL3_C0_PYTHON_BIN:?set ELAL3_C0_PYTHON_BIN}"
seed="${ELAL3_C0_SEED:-20260817}"
device="${ELAL3_C0_DEVICE:-cuda}"
expected_node="${ELAL3_C0_EXPECTED_NODE:-}"

[[ "${archive}" == /* && "${manifest}" == /* && "${output_root}" == /* && "${python_bin}" == /* ]] || fail "all paths must be absolute"
[[ "${archive_sha}" =~ ^[0-9a-f]{64}$ && "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid release SHA-256"
[[ "${seed}" =~ ^[0-9]+$ ]] || fail "seed must be an integer"
[[ "${device}" == cuda || "${device}" == cpu ]] || fail "device must be cuda or cpu"
[[ -f "${archive}" && ! -L "${archive}" && -f "${manifest}" && ! -L "${manifest}" ]] || fail "release files must be plain files"
[[ -x "${python_bin}" && -f "${python_bin}" && ! -L "${python_bin}" ]] || fail "Python must be a plain executable"
[[ "$(sha256sum "${archive}" | awk '{print $1}')" == "${archive_sha}" ]] || fail "archive SHA-256 differs"
[[ "$(sha256sum "${manifest}" | awk '{print $1}')" == "${manifest_sha}" ]] || fail "manifest SHA-256 differs"
[[ ! -e "${output_root}" && ! -L "${output_root}" ]] || fail "output root is not fresh"
[[ "${output_root}" != / && "${output_root##*/}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || fail "unsafe output root"
if [[ -n "${expected_node}" ]]; then
  [[ "$(hostname -s)" == "${expected_node}" ]] || fail "unexpected node: $(hostname -s)"
fi

scratch_parent="${SLURM_TMPDIR:-/tmp}"
[[ "${scratch_parent}" == /* && -d "${scratch_parent}" && ! -L "${scratch_parent}" && -w "${scratch_parent}" ]] || fail "invalid scratch parent"
scratch="$(mktemp -d "${scratch_parent%/}/elal3-c0.XXXXXX")"
cleanup() {
  local status=$?
  trap - EXIT
  case "${scratch}" in
    "${scratch_parent%/}/elal3-c0."*) ;;
    *) echo "[elal3-c0] refusing unsafe scratch cleanup" >&2; exit 2 ;;
  esac
  if [[ -d "${scratch}" && ! -L "${scratch}" ]]; then
    chmod -R u+w -- "${scratch}" || status=2
    rm -rf -- "${scratch}" || status=2
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

mkdir -p -- "$(dirname -- "${output_root}")"
mkdir -- "${output_root}"
chmod 0700 -- "${output_root}"
mkdir -- "${scratch}/source"

"${python_bin}" -I -B - "${archive}" "${manifest}" "${scratch}/source" <<'PY'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

archive_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
output = Path(sys.argv[3])
manifest_raw = manifest_path.read_bytes()
manifest = json.loads(manifest_raw)
canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
if canonical != manifest_raw:
    raise SystemExit("release manifest is not canonical")
if manifest.get("schema_version") != "bernini-elal3-synthetic-c0-release-v1":
    raise SystemExit("release schema differs")
if any(manifest.get(key) is not False for key in ("training_authorized", "exact160_authorized", "representation_semantics_qualified")):
    raise SystemExit("release authority boundary differs")
if hashlib.sha256(archive_path.read_bytes()).hexdigest() != manifest.get("archive_sha256"):
    raise SystemExit("archive/manifest binding differs")
expected = {row["path"]: row for row in manifest.get("files", [])}
if len(expected) != 3:
    raise SystemExit("release file closure differs")
with tarfile.open(archive_path, "r:") as release:
    members = release.getmembers()
    names = [member.name for member in members]
    if names != sorted(names, key=lambda value: value.encode("ascii")) or set(names) != set(expected):
        raise SystemExit("archive member order/closure differs")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not member.isreg() or member.mode != 0o444:
            raise SystemExit(f"unsafe archive member: {member.name}")
        raw = release.extractfile(member).read()
        row = expected[member.name]
        if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise SystemExit(f"archive payload differs: {member.name}")
        target = output.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o400)
        try:
            remaining = memoryview(raw)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise SystemExit(f"archive write made no progress: {member.name}")
                remaining = remaining[written:]
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
PY

method_root="${scratch}/source/methods/bernini_action_editing"
[[ -f "${method_root}/run_elal3_c0_v1.py" && -f "${method_root}/elal3_c0_v1.py" ]] || fail "extracted runtime closure differs"

run_arm() {
  local variant="$1" width="$2" label="$3" receipt
  receipt="${output_root}/${label}.json"
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${python_bin}" -B "${method_root}/run_elal3_c0_v1.py" \
      --variant "${variant}" \
      --attention-width "${width}" \
      --device "${device}" \
      --seed "${seed}" \
      --output "${receipt}"
  [[ -f "${receipt}" && ! -L "${receipt}" ]] || fail "missing receipt for ${label}"
}

run_arm no_relation 64 no_relation-w64
run_arm full 64 full-w64
run_arm full 128 full-w128

"${python_bin}" -I -B - "${output_root}" "${archive_sha}" "${manifest_sha}" "${seed}" "${device}" <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import sys

root = Path(sys.argv[1])
archive_sha, manifest_sha, seed, requested_device = sys.argv[2:6]
labels = ("no_relation-w64", "full-w64", "full-w128")
expected_arms = {
    "no_relation-w64": ("no_relation", 64, "SYNTHETIC_ABLATION_GO", False, 216),
    "full-w64": ("full", 64, "SYNTHETIC_C0_GO", True, 219),
    "full-w128": ("full", 128, "SYNTHETIC_C0_GO", True, 219),
}
receipt_keys = {
    "schema_version", "module_schema_version", "status", "variant",
    "attention_width", "registered_arm", "seed", "device", "torch_version",
    "torch_cuda_version", "torch_hip_version", "platform", "synthetic_harness",
    "synthetic_inputs_generated_before_arm_initialization", "synthetic_input_rows",
    "synthetic_input_digest", "paired_initialization_schema",
    "paired_master_plan_rows", "paired_master_plan_row_count",
    "paired_master_plan_digest", "paired_active_parameter_mapping",
    "paired_active_parameter_row_count", "paired_active_parameter_mapping_digest",
    "frozen_output_encoder_receipt", "real_bernini_checkpoint_loaded",
    "representation_tokenizer_loaded_or_qualified", "training_authorized",
    "scientific_promotion_authorized", "engineering_gate_pass",
    "synthetic_full_structure_gate_pass", "complete_elal3_c0",
    "production_elal3_c0_authority", "gates", "thresholds", "geometry",
    "intervention_target_rms_deltas_from_correct", "sp1_sp4_max_abs_difference",
    "padding_rows_observed_across_block_hooks", "output_action_loss",
    "renderer_block_grad_norms", "action_injection_block_grad_norms",
    "action_parameter_grad_norms", "action_latent_input_grad_norms",
    "parameter_estimate", "activation_estimate", "elapsed_seconds", "cuda_memory",
    "receipt_digest",
}
gate_keys = {
    "source_rows_bit_exact", "padding_rows_bit_exact",
    "all_30_blocks_hooked_per_forward", "sp4_matches_sp1",
    "zero_intervention_non_equivalent", "phase_reverse_non_equivalent",
    "role_slot_swap_non_equivalent", "relation_zero_non_equivalent",
    "renderer_and_action_all_30_gradients_finite_nonzero",
    "all_active_action_parameters_gradients_finite_nonzero",
    "required_action_latent_inputs_gradients_finite_nonzero",
    "q_camera_nuisance_is_not_injected_into_action_loss",
    "frozen_output_encoder_parameters_have_no_gradient",
    "checkpoint_route_identity_replays", "output_action_loss_finite",
}
input_keys = {
    "q_local", "q_entity", "q_relation", "q_phase", "q_terminal", "q_camera",
    "entity_presence", "temporal_valid", "relation_valid", "phase_valid", "hidden",
}

def reject(message):
    raise SystemExit("ELAL-3 paired summary rejected: " + message)

def canonical(value):
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        reject("non-canonical JSON value: " + str(error))

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def require(condition, message):
    if not condition:
        reject(message)

def stable_receipt(path):
    before_name = path.lstat()
    require(stat.S_ISREG(before_name.st_mode) and not path.is_symlink(), f"non-plain receipt {path.name}")
    require(stat.S_IMODE(before_name.st_mode) == 0o640 and before_name.st_nlink == 1, f"receipt mode/link differs {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        first = bytearray()
        while len(first) < before.st_size:
            block = os.read(descriptor, before.st_size - len(first))
            require(bool(block), f"receipt truncated {path.name}")
            first.extend(block)
        require(os.read(descriptor, 1) == b"", f"receipt grew {path.name}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        second = bytearray()
        while len(second) < before.st_size:
            block = os.read(descriptor, before.st_size - len(second))
            require(bool(block), f"receipt replay truncated {path.name}")
            second.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_name = path.lstat()
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_uid,
        item.st_gid, item.st_rdev, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
    )
    require(identity(before_name) == identity(before) == identity(after) == identity(after_name), f"receipt identity drift {path.name}")
    require(first == second, f"receipt bytes drift {path.name}")
    return bytes(first)

def validate_tensor_row(row, label):
    require(type(row) is dict and set(row) == {"shape", "dtype", "sha256"}, f"tensor row closure differs {label}")
    require(type(row["shape"]) is list and all(type(value) is int and value >= 0 for value in row["shape"]), f"tensor shape differs {label}")
    require(type(row["dtype"]) is str and row["dtype"].startswith("torch."), f"tensor dtype differs {label}")
    require(type(row["sha256"]) is str and len(row["sha256"]) == 64, f"tensor SHA differs {label}")

def validate_gradient_rows(value, label):
    require(type(value) is list and len(value) == 30, f"gradient row count differs {label}")
    for index, row in enumerate(value):
        require(type(row) is dict and set(row) == {"block_index", "grad_norm"}, f"gradient row closure differs {label}")
        require(row["block_index"] == index, f"gradient index differs {label}")
        require(type(row["grad_norm"]) in (int, float) and math.isfinite(row["grad_norm"]) and row["grad_norm"] > 0.0, f"gradient value differs {label}")

rows = []
receipts = {}
for label in labels:
    path = root / f"{label}.json"
    raw = stable_receipt(path)
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        reject(f"receipt parse failed {label}: {error}")
    require(raw == canonical(value) + b"\n", f"receipt is not canonical {label}")
    require(type(value) is dict and set(value) == receipt_keys, f"receipt schema closure differs {label}")
    unsigned = {key: item for key, item in value.items() if key != "receipt_digest"}
    require(value["receipt_digest"] == digest(unsigned), f"receipt self-digest differs {label}")
    variant, width, status, full_gate, active_count = expected_arms[label]
    require(value["schema_version"] == "bernini-elal3-synthetic-c0-receipt-v1", f"receipt version differs {label}")
    require(value["paired_initialization_schema"] == "bernini-elal3-c0-paired-initialization-v1", f"paired schema differs {label}")
    require(value["registered_arm"] == label and value["variant"] == variant and value["attention_width"] == width, f"arm identity differs {label}")
    require(value["status"] == status and value["seed"] == int(seed), f"status/seed differs {label}")
    require(value["device"] == requested_device, f"device differs {label}")
    for key in (
        "synthetic_harness", "synthetic_inputs_generated_before_arm_initialization",
        "engineering_gate_pass",
    ):
        require(value[key] is True, f"required true authority differs {label}:{key}")
    for key in (
        "real_bernini_checkpoint_loaded", "representation_tokenizer_loaded_or_qualified",
        "training_authorized", "scientific_promotion_authorized", "complete_elal3_c0",
        "production_elal3_c0_authority",
    ):
        require(value[key] is False, f"required false authority differs {label}:{key}")
    require(value["synthetic_full_structure_gate_pass"] is full_gate, f"full structure gate differs {label}")
    require(type(value["gates"]) is dict and set(value["gates"]) == gate_keys and all(item is True for item in value["gates"].values()), f"engineering gates differ {label}")
    require(type(value["synthetic_input_rows"]) is dict and set(value["synthetic_input_rows"]) == input_keys, f"input closure differs {label}")
    for name, tensor_row in value["synthetic_input_rows"].items():
        validate_tensor_row(tensor_row, f"{label}:{name}")
    require(value["synthetic_input_digest"] == digest(value["synthetic_input_rows"]), f"input digest differs {label}")
    master_rows = value["paired_master_plan_rows"]
    require(type(master_rows) is list and len(master_rows) == value["paired_master_plan_row_count"] == 219, f"master plan count differs {label}")
    require(value["paired_master_plan_digest"] == digest(master_rows), f"master plan digest differs {label}")
    namespaces = []
    master_by_namespace = {}
    for master_row in master_rows:
        require(type(master_row) is dict and set(master_row) == {"namespace", "shape", "dtype", "initializer", "sha256"}, f"master row closure differs {label}")
        namespace = master_row["namespace"]
        require(type(namespace) is str and namespace and namespace not in master_by_namespace, f"master namespace differs {label}")
        require(master_row["dtype"] == "torch.float32" and type(master_row["shape"]) is list, f"master ABI differs {label}:{namespace}")
        require(type(master_row["initializer"]) is dict and type(master_row["sha256"]) is str and len(master_row["sha256"]) == 64, f"master evidence differs {label}:{namespace}")
        namespaces.append(namespace)
        master_by_namespace[namespace] = master_row
    require(namespaces == sorted(namespaces), f"master rows are not sorted {label}")
    frozen = value["frozen_output_encoder_receipt"]
    require(type(frozen) is dict and set(frozen) == {"namespace", "master_sha256", "shape", "dtype", "sha256", "requires_grad"}, f"frozen receipt closure differs {label}")
    require(frozen["namespace"] == "frozen_output_encoder.weight" and frozen["requires_grad"] is False, f"frozen receipt semantics differ {label}")
    frozen_master = master_by_namespace[frozen["namespace"]]
    require(frozen["master_sha256"] == frozen_master["sha256"] == frozen["sha256"] and frozen["shape"] == frozen_master["shape"] and frozen["dtype"] == frozen_master["dtype"], f"frozen readout binding differs {label}")
    active_rows = value["paired_active_parameter_mapping"]
    require(type(active_rows) is list and len(active_rows) == value["paired_active_parameter_row_count"] == active_count, f"active mapping count differs {label}")
    require(value["paired_active_parameter_mapping_digest"] == digest(active_rows), f"active mapping digest differs {label}")
    active_keys = set()
    for active in active_rows:
        require(type(active) is dict and set(active) == {"component", "parameter", "master_namespace", "master_sha256", "master_shape", "active_slice", "active_shape", "active_dtype", "active_sha256", "requires_grad"}, f"active mapping closure differs {label}")
        active_key = (active["component"], active["parameter"])
        require(active_key not in active_keys, f"duplicate active mapping {label}:{active_key}")
        active_keys.add(active_key)
        master = master_by_namespace.get(active["master_namespace"])
        require(master is not None and active["master_sha256"] == master["sha256"] and active["master_shape"] == master["shape"], f"active/master binding differs {label}:{active_key}")
        require(active["active_dtype"] == "torch.float32" and type(active["active_sha256"]) is str and len(active["active_sha256"]) == 64, f"active tensor evidence differs {label}:{active_key}")
        parameter = active["parameter"]
        expected_slice = {"kind": "all"}
        if parameter.endswith((".query.weight", ".key.weight", ".value.weight")):
            expected_slice = {"kind": "prefix_rows", "start": 0, "stop": width}
        elif parameter.endswith(".output.weight"):
            expected_slice = {"kind": "prefix_columns", "start": 0, "stop": width}
        require(active["active_slice"] == expected_slice, f"active slice differs {label}:{parameter}")
    validate_gradient_rows(value["renderer_block_grad_norms"], f"{label}:renderer")
    validate_gradient_rows(value["action_injection_block_grad_norms"], f"{label}:action")
    receipts[label] = value
    rows.append({"label": label, "path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw), "receipt_digest": value.get("receipt_digest")})

reference = receipts["full-w64"]
for label in labels:
    value = receipts[label]
    require(value["synthetic_input_rows"] == reference["synthetic_input_rows"], f"paired inputs differ {label}")
    require(value["synthetic_input_digest"] == reference["synthetic_input_digest"], f"paired input digest differs {label}")
    require(value["paired_master_plan_rows"] == reference["paired_master_plan_rows"], f"paired master plan differs {label}")
    require(value["paired_master_plan_digest"] == reference["paired_master_plan_digest"], f"paired master digest differs {label}")
    require(value["frozen_output_encoder_receipt"] == reference["frozen_output_encoder_receipt"], f"paired frozen readout differs {label}")

def active_map(value):
    return {
        (row["component"], row["parameter"]): row
        for row in value["paired_active_parameter_mapping"]
    }

no_relation = active_map(receipts["no_relation-w64"])
full64 = active_map(receipts["full-w64"])
full128 = active_map(receipts["full-w128"])
for key in set(no_relation) & set(full64):
    require(no_relation[key] == full64[key], f"common w64 parameter differs {key}")
for key in set(full64) & set(full128):
    left, right = full64[key], full128[key]
    for field in ("component", "parameter", "master_namespace", "master_sha256", "master_shape", "requires_grad"):
        require(left[field] == right[field], f"w64/w128 master mapping differs {key}:{field}")
    if left["active_slice"] == {"kind": "all"}:
        require(right["active_slice"] == {"kind": "all"} and left["active_sha256"] == right["active_sha256"], f"w64/w128 shared tensor differs {key}")

summary = {
    "schema_version": "bernini-elal3-c0-node-run-v2",
    "status": "SYNTHETIC_PAIRED_C0_ONLY_NO_TRAINING_AUTHORITY",
    "hostname": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
    "source_archive_sha256": archive_sha,
    "source_manifest_sha256": manifest_sha,
    "seed": int(seed),
    "arms": rows,
    "paired_comparison_validated": True,
    "paired_initialization_schema": reference["paired_initialization_schema"],
    "synthetic_input_digest": reference["synthetic_input_digest"],
    "paired_master_plan_digest": reference["paired_master_plan_digest"],
    "frozen_output_encoder_receipt": reference["frozen_output_encoder_receipt"],
    "training_authorized": False,
    "exact160_authorized": False,
    "scientific_promotion_authorized": False,
}
summary["summary_digest"] = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
raw = json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
target = root / "RUN_COMPLETE.json"
descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
try:
    remaining = memoryview(raw)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise SystemExit("summary write made no progress")
        remaining = remaining[written:]
    os.fchmod(descriptor, 0o444)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY

chmod 0444 -- "${output_root}"/*.json
chmod 0555 -- "${output_root}"
echo "[elal3-c0] PASS node=$(hostname -s) output=${output_root} synthetic_only=true"

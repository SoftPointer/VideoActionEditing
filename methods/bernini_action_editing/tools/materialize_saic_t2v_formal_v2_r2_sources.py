#!/usr/bin/env python3
"""Materialize the formal-v2-r2 source cascade from one admitted WORLD8 proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
PLACEHOLDER = re.compile(rb"__R2_[A-Z0-9_]+__")
EXPECTED_ADMISSION_SHA256 = (
    "4da203469b022fcdcf7a4d6ba377769d6b220e5b6ce0066e60587104f8405e26"
)
EXPECTED_ADMISSION_DIGEST = (
    "616c1a7587679975329e1211653720a7a2726c7cd676bcb0eedeea4bdda7b50d"
)
EXPECTED_ADMISSION_PATH = (
    "/vast/users/guangyi.chen/causal_group/peiyuan.zhu/work/VideoEditing/"
    "VideoEdit_experiments/bernini_saic_v1_20260809/canaries/"
    "saic-formal-v2-retained-fd-world8-canary-3571e921-6a83aaec-r7/"
    "canary-admission.json"
)
EXPECTED_JOB_ID = "134908"
EXPECTED_PROBE_SHA256 = (
    "d51ebf1f894d63483943042faaa2c6ccbf812c0f93769980084bae72f8ab84d8"
)
EXPECTED_PROBE_DIGEST = (
    "a37e44c12f935a4f4e11ab08364b019799809fd13299af6459ca6797a8333fb7"
)
EXPECTED_TEMPLATES = {
    "gate": "677c8021eb7ae1c7eae5309bc2e732a8934416a234ddd6ba9d8b7f2ba47dfee0",
    "wrapper": "16b8625f6e5620102e8800cdffd939346ffeb0a77618fbbf5ef0b6fa8eb80e5b",
    "submitter": "e35e6595bd042ca048483ccabd4c215e2b5427607aa5f8264a4284053d67b514",
    "pins": "1a01dbc08572b14a7d293f476db65b9e9ed1baa17735912f424451b1a8abcd51",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_exact(path: Path, expected_sha: str, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} is not a plain file: {path}")
    payload = path.read_bytes()
    if sha_bytes(payload) != expected_sha:
        raise SystemExit(f"{label} SHA-256 differs")
    return payload


def decode_admission(path: Path) -> Mapping[str, Any]:
    raw = read_exact(path, EXPECTED_ADMISSION_SHA256, "retained-FD admission")
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"retained-FD admission encoding differs: {error}")
    if not isinstance(value, dict) or raw != canonical(value) + b"\n":
        raise SystemExit("retained-FD admission is not canonical")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if digest != EXPECTED_ADMISSION_DIGEST or digest != sha_bytes(canonical(unsigned)):
        raise SystemExit("retained-FD admission seal differs")
    authority = value.get("authority")
    binding = value.get("probe_admission_binding")
    observation = value.get("sacct_terminal_observation")
    row = observation.get("parsed_row") if isinstance(observation, dict) else None
    compute = binding.get("compute_bash") if isinstance(binding, dict) else None
    if (
        value.get("schema_version")
        != "saic-formal-v2-retained-fd-world8-canary-admission-v1"
        or value.get("status")
        != "terminal_completed_retained_fd_world8_operational_admitted"
        or str(value.get("job_id")) != EXPECTED_JOB_ID
        or value.get("job_success") is not True
        or value.get("operational_canary_admitted") is not True
        or value.get("underlying_world8_closure_deep_validated") is not True
        or value.get("science_generation_entered") is not False
        or value.get("formal_admission") is not False
        or value.get("external_formal_submitter_exact_pin_required") is not True
        or not isinstance(authority, dict)
        or any(authority.get(key) is not False for key in (
            "scientific", "generation", "training", "publication",
            "formal_job_authorized", "authorizes_formal_submission_by_itself",
            "reusable_for_other_release",
        ))
        or not isinstance(binding, dict)
        or binding.get("sha256") != EXPECTED_PROBE_SHA256
        or binding.get("receipt_digest") != EXPECTED_PROBE_DIGEST
        or not isinstance(compute, dict)
        or compute.get("path") != "/usr/bin/bash"
        or compute.get("retained_fd_survives_bash_script_handoff") is not True
        or compute.get("brace_fd_redirection_supported") is not True
        or compute.get("varredir_close_option_required") is not False
        or not isinstance(observation, dict)
        or observation.get("returncode") != 0
        or observation.get("exact_single_row") is not True
        or observation.get("exact_submit_line") is not True
        or not isinstance(row, dict)
        or row.get("State") != "COMPLETED"
        or row.get("ExitCode") != "0:0"
        or row.get("JobIDRaw") != EXPECTED_JOB_ID
        or "/proc/self/fd/" not in row.get("SubmitLine", "")
    ):
        raise SystemExit("retained-FD admission semantics differ")
    return value


def replace(source: bytes, values: Mapping[bytes, bytes], label: str) -> bytes:
    result = source
    for token, value in values.items():
        count = result.count(token)
        if count:
            result = result.replace(token, value)
    unresolved = sorted(set(PLACEHOLDER.findall(result)))
    if unresolved:
        raise SystemExit(f"{label} retains placeholders: {unresolved!r}")
    return result


def write_fresh(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644
    )
    try:
        offset = 0
        while offset < len(payload):
            wrote = os.write(descriptor, payload[offset:])
            if wrote <= 0:
                raise SystemExit(f"write stalled: {path}")
            offset += wrote
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", required=True)
    args = parser.parse_args(argv)
    method_root = Path(__file__).resolve().parents[1]
    templates = {
        "gate": method_root / "saic_t2v_formal_v2_r2_gate_template.py",
        "wrapper": method_root / "scripts/auh_generate_saic_pure_t2v_event_bank_topup_all8_v5_r2_template.sbatch",
        "submitter": method_root / "tools/submit_saic_t2v_topup_r6_v3_r2_template.py",
        "pins": method_root / "tools/saic_t2v_formal_v2_r2_release_pins.template.txt",
    }
    outputs = {
        "gate": method_root / "saic_t2v_formal_v2_r2_gate.py",
        "wrapper": method_root / "scripts/auh_generate_saic_pure_t2v_event_bank_topup_all8_v5.sbatch",
        "submitter": method_root / "tools/submit_saic_t2v_topup_r6_v3.py",
        "pins": method_root / "tools/saic_t2v_formal_v2_r2_release_pins.txt",
    }
    sources = {
        label: read_exact(path, EXPECTED_TEMPLATES[label], f"{label} template")
        for label, path in templates.items()
    }
    admission = decode_admission(Path(args.admission))
    observation = admission["sacct_terminal_observation"]
    row = observation["parsed_row"]
    compute = admission["probe_admission_binding"]["compute_bash"]
    raw_values = {
        "__R2_RETAINED_FD_CANARY_JOB_ID__": EXPECTED_JOB_ID,
        "__R2_RETAINED_FD_CANARY_ADMISSION_PATH__": EXPECTED_ADMISSION_PATH,
        "__R2_RETAINED_FD_CANARY_ADMISSION_SHA256__": EXPECTED_ADMISSION_SHA256,
        "__R2_RETAINED_FD_CANARY_ADMISSION_DIGEST__": EXPECTED_ADMISSION_DIGEST,
        "__R2_RETAINED_FD_CANARY_SACCT_STDOUT_SHA256__": observation["stdout_sha256"],
        "__R2_RETAINED_FD_CANARY_NODELIST__": row["NodeList"],
        "__R2_RETAINED_FD_CANARY_START__": row["Start"],
        "__R2_RETAINED_FD_CANARY_END__": row["End"],
        "__R2_RETAINED_FD_CANARY_ELAPSED__": row["Elapsed"],
        "__R2_RETAINED_FD_CANARY_SUBMIT_LINE__": row["SubmitLine"],
        "__R2_COMPUTE_BASH_SHA256__": compute["sha256"],
        "__R2_COMPUTE_BASH_VERSION_STDOUT_SHA256__": compute["version_stdout_sha256"],
        "__R2_COMPUTE_BASH_VERSION_FIRST_LINE__": compute["version_first_line"],
    }
    values = {key.encode("ascii"): value.encode("ascii") for key, value in raw_values.items()}
    gate = replace(sources["gate"], values, "gate")
    gate_sha = sha_bytes(gate)
    cascade = dict(values)
    cascade[b"__R2_FORMAL_GATE_SHA256__"] = gate_sha.encode("ascii")
    wrapper = replace(sources["wrapper"], cascade, "wrapper")
    wrapper_sha = sha_bytes(wrapper)
    cascade[b"__R2_FORMAL_WRAPPER_SHA256__"] = wrapper_sha.encode("ascii")
    submitter = replace(sources["submitter"], cascade, "submitter")
    submitter_sha = sha_bytes(submitter)
    cascade[b"__R2_FORMAL_SUBMITTER_SHA256__"] = submitter_sha.encode("ascii")
    pins = replace(sources["pins"], cascade, "pin manifest")
    for label, payload in (
        ("gate", gate), ("wrapper", wrapper), ("submitter", submitter), ("pins", pins)
    ):
        write_fresh(outputs[label], payload)
    print(json.dumps({
        "gate_sha256": gate_sha,
        "wrapper_sha256": wrapper_sha,
        "submitter_sha256": submitter_sha,
        "admission_sha256": EXPECTED_ADMISSION_SHA256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

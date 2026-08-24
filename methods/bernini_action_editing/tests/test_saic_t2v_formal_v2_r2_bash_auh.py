#!/usr/bin/env python3
"""AUH-only hostile/static checks for the formal-r2 Bash handoff template.

This file is intentionally not a local unit test.  A fresh AUH staging copy
must substitute the terminal replacement-canary pins before its release-mode
checks may pass.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess


PLACEHOLDER = re.compile(r"__R2_[A-Z0-9_]+__")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--wrapper", required=True)
    parser.add_argument("--submitter", required=True)
    parser.add_argument("--expected-bash-sha256", required=True)
    parser.add_argument("--expected-bash-version-stdout-sha256", required=True)
    parser.add_argument("--expected-bash-version-first-line", required=True)
    parser.add_argument("--allow-template-placeholders", action="store_true")
    args = parser.parse_args()

    gate = Path(args.gate).read_text(encoding="ascii")
    wrapper = Path(args.wrapper).read_text(encoding="ascii")
    submitter = Path(args.submitter).read_text(encoding="ascii")
    combined = gate + wrapper + submitter
    if not args.allow_template_placeholders and PLACEHOLDER.search(combined):
        raise SystemExit("formal r2 release retains placeholder pins")
    if "shopt -u varredir_close" in wrapper:
        raise SystemExit("formal r2 wrapper retains unsupported varredir_close probe")
    if not wrapper.startswith("#!/usr/bin/bash\n"):
        raise SystemExit("formal r2 wrapper is not bound to direct /usr/bin/bash")
    if "#!/usr/bin/env bash" in wrapper:
        raise SystemExit("formal r2 wrapper permits PATH-selected Bash")
    required_wrapper = (
        '[[ "${BASH}" == "${expected_compute_bash}"',
        'realpath -e -- "/proc/$$/exe"',
        'fail "current formal Bash interpreter differs"',
        'exec {gate_fd}<"${gate}"',
        'exec {effective_fd}<"${effective_launcher}"',
        'exec {guard_fd}<"${guard}"',
        'exec {probe_validator_fd}<"${probe_validator}"',
        'SAIC_T2V_FV2_PROBE_VALIDATOR_FD_PATH',
        '--retained-fd-canary-admission "${retained_fd_canary_admission}"',
        '--compute-bash-probe-admission "${compute_bash_probe_admission}"',
        'exec "${compute_bash}" "${effective_fd_path}"',
        "check_compute_bash_contract",
    )
    if any(item not in wrapper for item in required_wrapper):
        raise SystemExit("formal r2 retained-FD wrapper closure differs")
    for item in (
        "validate_retained_fd_canary",
        'observe_retained_fd_canary_sacct("runtime_gate_after_formal_sbatch")',
        "validate_compute_bash",
        '"three_independent_operational_proof_objects_required": True',
        'value.get("probe_admission_binding") != probe_binding',
        'value.get("probe_validator_sha256") != PROBE_VALIDATOR_SHA256',
        '"SubmitLine%8192"',
        '"submit_line_sha256"',
        '"retained_wrapper_fd"',
        '"exact_submit_line"',
    ):
        if item not in gate:
            raise SystemExit("formal r2 three-gate closure differs")
    # Final r2 materialization must upgrade this template to three independent
    # proof objects: exact60 lifecycle, sealed Bash probe, and WORLD8 transport
    # binding that exact probe.  A combined WORLD8 compute_bash claim is not a
    # substitute for the standalone probe admission.
    if not args.allow_template_placeholders:
        for item in (
            "validate_compute_bash_probe_admission",
            "compute_bash_probe_admission",
            "exact60_lifecycle_probe_world8_transport_non_substitutability",
            '"three_independent_operational_proof_objects_required": True',
        ):
            if item not in combined:
                raise SystemExit("formal r2 three-proof cascade differs")
    if any(item not in submitter for item in (
        "gate_module.validate_compute_bash_probe_admission",
        "gate_module.validate_retained_fd_canary",
        '"submitter_before_formal_sbatch"',
        '"hold": False',
        '"dependency": None',
    )):
        raise SystemExit("formal r2 submitter closure differs")
    forbidden_transport = "SAIC_T2V_FV2_COMPUTE_BASH_VERSION_FIRST_LINE"
    if forbidden_transport in gate or forbidden_transport in submitter:
        raise SystemExit("comma-bearing Bash first line leaked into Slurm exports")
    if 'compute_bash_first_line="${expected_compute_bash_first_line}"' not in wrapper:
        raise SystemExit("formal wrapper lacks embedded Bash first-line pin")

    bash = Path("/usr/bin/bash")
    if not bash.is_file() or bash.is_symlink():
        raise SystemExit("AUH /usr/bin/bash identity differs")
    if sha256(bash) != args.expected_bash_sha256:
        raise SystemExit("AUH /usr/bin/bash SHA differs")
    completed = subprocess.run(
        [str(bash), "--version"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=60, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    lines = completed.stdout.decode("ascii").splitlines()
    if (
        completed.returncode != 0
        or completed.stderr != b""
        or hashlib.sha256(completed.stdout).hexdigest()
        != args.expected_bash_version_stdout_sha256
        or not lines
        or lines[0] != args.expected_bash_version_first_line
    ):
        raise SystemExit("AUH Bash version contract differs")

    completed = subprocess.run(
        [str(bash), "-n", args.wrapper], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        timeout=60, env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if completed.returncode != 0 or completed.stdout or completed.stderr:
        raise SystemExit("AUH Bash parser rejected formal r2 wrapper")
    print("SAIC_FORMAL_V2_R2_AUH_BASH_STATIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

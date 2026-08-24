#!/usr/bin/env python3
"""AUH-only hostile/static checks for the top-up v2 review release.

This source never builds media and never submits a job.  The template-mode flag
permits unresolved pins for staging review; final release mode rejects every
placeholder and verifies the exact admitted AUH Bash identity before parsing
the launcher with that Bash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any


PLACEHOLDER = re.compile(r"__REVIEW_[A-Z0-9_]+__")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SystemExit(f"review release manifest duplicate key: {key!r}")
        value[key] = item
    return value


def plain_file(path: Path, expected_sha256: str, *, executable: bool = False) -> None:
    info = path.lstat()
    if (
        path.resolve(strict=True) != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (not executable and info.st_uid != os.getuid())
        or stat.S_IMODE(info.st_mode) & 0o022
        or (not executable and stat.S_IMODE(info.st_mode) != 0o444)
        or (executable and not os.access(path, os.X_OK))
        or sha256(path) != expected_sha256
    ):
        raise SystemExit(f"review release file identity differs: {path}")


def validate_release_manifest(args: argparse.Namespace) -> None:
    required = (
        args.release_manifest,
        args.expected_release_manifest_sha256,
        args.expected_release_manifest_digest,
        args.hostile,
        args.manifest_materializer,
        args.source_archive,
    )
    if any(value is None for value in required):
        raise SystemExit("review release manifest arguments are incomplete")
    manifest_path = Path(args.release_manifest)
    plain_file(manifest_path, args.expected_release_manifest_sha256)
    raw = manifest_path.read_bytes()
    try:
        value = json.loads(
            raw.decode("ascii"), object_pairs_hook=no_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot decode review release manifest: {error}")
    if type(value) is not dict or raw != canonical(value) + b"\n":
        raise SystemExit("review release manifest is not canonical JSON")
    unsigned = dict(value)
    digest = unsigned.pop("receipt_digest", None)
    if (
        digest != args.expected_release_manifest_digest
        or not re.fullmatch(r"[0-9a-f]{64}", str(digest))
        or hashlib.sha256(canonical(unsigned)).hexdigest() != digest
    ):
        raise SystemExit("review release manifest seal differs")
    release_root = manifest_path.parent
    inputs_root = release_root / "inputs"
    expected_inputs = {
        "manifest_materializer": Path(args.manifest_materializer),
        "adapter": Path(args.adapter), "launcher": Path(args.launcher),
        "submitter": Path(args.submitter), "postflight": Path(args.postflight),
        "hostile": Path(args.hostile),
        "source_archive": Path(args.source_archive),
    }
    expected_executables = {
        "python": Path(
            "/vast/users/guangyi.chen/anaconda3/envs/vace/bin/python3.12"
        ),
        "ffmpeg": Path(
            "/vast/users/guangyi.chen/anaconda3/envs/vace/lib/python3.12/"
            "site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        ),
        "compute_bash": Path("/usr/bin/bash"),
        "sacct": Path("/usr/bin/sacct"),
    }
    inputs = value.get("inputs")
    formal_inputs = value.get("formal_inputs")
    executables = value.get("executables")
    if (
        set(value) != {
            "schema_version", "status", "release_root", "inputs",
            "formal_inputs", "executables", "authority", "receipt_digest",
        }
        or value.get("schema_version")
        != "saic-t2v-topup-review-v2-release-manifest-v1"
        or value.get("status") != "sealed_before_review_submission"
        or value.get("release_root") != str(release_root)
        or value.get("authority") != {
            "scientific": False, "human_review": False,
            "event_verified": False, "identity_preservation_verified": False,
            "candidate_selection": False, "seed_selection": False,
            "training_target": False, "training": False,
            "optimizer_step": False, "parameter_update": False,
        }
        or type(inputs) is not dict
        or set(inputs) != set(expected_inputs)
        or type(formal_inputs) is not dict
        or set(formal_inputs) != {"master_receipt", "submission_receipt"}
        or type(executables) is not dict
        or set(executables) != set(expected_executables)
        or set(inputs_root.iterdir()) != set(expected_inputs.values())
        or set(release_root.iterdir()) != {inputs_root, manifest_path}
    ):
        raise SystemExit("review release manifest closure differs")
    for name, path in expected_inputs.items():
        binding = inputs[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256"))) is None
        ):
            raise SystemExit(f"review release {name} binding differs")
        plain_file(path, binding["sha256"])
    for name, binding in formal_inputs.items():
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256"))) is None
        ):
            raise SystemExit(f"review release formal {name} binding differs")
        plain_file(Path(binding["path"]), binding["sha256"])
    for name, path in expected_executables.items():
        binding = executables[name]
        if (
            type(binding) is not dict
            or set(binding) != {"path", "sha256"}
            or binding.get("path") != str(path)
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256"))) is None
        ):
            raise SystemExit(f"review release executable {name} binding differs")
        plain_file(path, binding["sha256"], executable=True)
    if (
        Path(args.hostile) != Path(__file__)
        or Path(args.hostile).resolve(strict=True)
        != Path(__file__).resolve(strict=True)
    ):
        raise SystemExit("review hostile execution path differs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--submitter", required=True)
    parser.add_argument("--postflight", required=True)
    parser.add_argument("--hostile")
    parser.add_argument("--manifest-materializer")
    parser.add_argument("--source-archive")
    parser.add_argument("--release-manifest")
    parser.add_argument("--expected-release-manifest-sha256")
    parser.add_argument("--expected-release-manifest-digest")
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--expected-postflight-sha256", required=True)
    parser.add_argument("--expected-bash-sha256", required=True)
    parser.add_argument("--expected-bash-version-stdout-sha256", required=True)
    parser.add_argument("--expected-bash-version-first-line", required=True)
    parser.add_argument("--allow-template-placeholders", action="store_true")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    launcher_path = Path(args.launcher)
    submitter_path = Path(args.submitter)
    postflight_path = Path(args.postflight)
    adapter = adapter_path.read_text(encoding="ascii")
    launcher = launcher_path.read_text(encoding="ascii")
    submitter = submitter_path.read_text(encoding="ascii")
    postflight = postflight_path.read_text(encoding="ascii")
    hostile = Path(args.hostile).read_text(encoding="ascii") if args.hostile else ""
    materializer = (
        Path(args.manifest_materializer).read_text(encoding="ascii")
        if args.manifest_materializer else ""
    )
    combined = adapter + launcher + submitter + postflight + hostile + materializer
    if not args.allow_template_placeholders and PLACEHOLDER.search(combined):
        raise SystemExit("review release retains placeholder pins")
    if sha256(adapter_path) != args.expected_adapter_sha256:
        raise SystemExit("review adapter SHA differs")
    if sha256(launcher_path) != args.expected_launcher_sha256:
        raise SystemExit("review launcher SHA differs")
    if sha256(postflight_path) != args.expected_postflight_sha256:
        raise SystemExit("review postflight SHA differs")
    if not args.allow_template_placeholders:
        validate_release_manifest(args)
    if args.allow_template_placeholders:
        placeholder_postflight_pin = (
            "__" + "REVIEW_POSTFLIGHT_SHA256" + "__"
        )
        expected_postflight_literal = (
            'EXPECTED_POSTFLIGHT_SHA256 = (\n'
            f'    "{placeholder_postflight_pin}"\n'
            ')'
        )
    else:
        expected_postflight_literal = (
            'EXPECTED_POSTFLIGHT_SHA256 = (\n'
            f'    "{args.expected_postflight_sha256}"\n'
            ')'
        )
    if expected_postflight_literal not in submitter:
        raise SystemExit("review submitter does not pin hostile-audited postflight")
    for name, source in (
        ("adapter", adapter),
        ("submitter", submitter),
        ("postflight", postflight),
        ("hostile", hostile),
        ("manifest_materializer", materializer),
    ):
        if source:
            compile(source, name, "exec")

    if "shopt -u varredir_close" in launcher:
        raise SystemExit("review launcher uses unsupported varredir_close option")
    if not launcher.startswith("#!/usr/bin/bash\n"):
        raise SystemExit("review launcher is not bound to direct /usr/bin/bash")
    if "#!/usr/bin/env bash" in launcher:
        raise SystemExit("review launcher permits PATH-selected Bash")
    transport_sources = adapter + launcher + submitter + postflight + materializer
    if "SAIC_T2V_TOPUP_REVIEW_COMPUTE_BASH_VERSION_FIRST_LINE" in transport_sources:
        raise SystemExit("comma-bearing Bash first line leaked into Slurm exports")
    launcher_anchors = (
        "#SBATCH --cpus-per-task=32",
        "#SBATCH --mem=192G",
        "#SBATCH --time=08:00:00",
        'compute_bash_first_line="${expected_compute_bash_first_line}"',
        '[[ "${BASH}" == "${expected_compute_bash}" ]]',
        'realpath -e -- "/proc/$$/exe"',
        '"${runtime_adapter}" build',
        '"${runtime_adapter}" validate',
        '"${output_root}/index.html"',
        '"${output_root}/blind-review.html"',
        '"${automation_receipt}"',
        '"human_label_ingest_ready": False',
        '"machine_ingest_ready": False',
        '"blind_stage_public_surface_only": True',
        '"technical_html_may_publish_before_two_external_human_seals": False',
        '"assessor_private_mapping_may_be_copied_to_stage1": False',
        '"terminal_machine_ingest_scope": "assessor_private_only"',
        '"human_visible_machine_backfill_ready": False',
        '"assessor_private_machine_ingest_may_precede_human_seals": True',
        '"external_terminal_postflight_required": True',
        'response_fields = {',
        'or row.get("review_item_id") != expected_id',
        'or any(row[field] is not None for field in response_fields - {"review_item_id"})',
        "os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW",
        "dir_fd=parent_descriptor",
        "automation receipt same-FD reread differs",
        "sealed automation receipt identity differs",
    )
    if any(anchor not in launcher for anchor in launcher_anchors):
        raise SystemExit("review launcher closure differs")
    submitter_anchors = (
        '"hold": False',
        '"dependency": None',
        '"human_labels_may_be_autofilled": False',
        '"external_review_job_terminal_postflight_required": True',
        '"launcher_submitted_from_retained_fd": True',
        '"formal_terminal_input_bundle": {',
        '"bundle_is_not_a_separate_formal_terminal_admission": True',
        "validate_formal_three_gate_bundle(submission)",
        '"formal three-gate admission binding differs"',
        '"SubmitLine%8192"',
        '"submit_line_sha256"',
        '"retained_wrapper_fd"',
        '"exact_submit_line"',
        'external_observation.get("submit_line_sha256")',
        'external_observation.get("retained_wrapper_fd")',
        'external_observation.get("exact_submit_line")',
        '"review_submitter_after_formal_terminal_before_only_sbatch"',
        '"--export=NONE,"',
        'f"/proc/self/fd/{launcher_descriptor}"',
        '"review sealed submission receipt differs"',
    )
    if any(anchor not in submitter for anchor in submitter_anchors):
        raise SystemExit("review submitter closure differs")
    adapter_anchors = (
        'BRANCH_ORDER = tuple(topup.BRANCH_ORDER)',
        '_BLIND_ALIAS_DOMAIN = b"saic-topup-review-v2-r1-opaque-alias\\0"',
        "nonce = secrets.token_bytes(_BLIND_ALIAS_NONCE_BYTES)",
        'row["assessor_private_candidate_id"] = original_candidate_id',
        'row["assessor_private_source_row_id"] = original_row_id',
        'row["candidate_id"] = f"candidate-{alias_index:04d}"',
        'row["row_id"] = source_alias',
        'base._render_blind_html = _render_blind_html',
        '"camera_only": "Counterfactual camera-only negative: "',
        'instruction = instruction[len(prefix):]',
        'assessor_private_ids = {',
        '"incomplete", "camera_only", "appearance_only"',
        'f"s{item[\'seed\']}" for item in items',
        '"href=" not in lower and "seed" not in lower',
        '"v2 blind HTML bytes differ from opaque renderer"',
        '"v2 opaque artifact namespace differs"',
        '"incomplete"',
        '"camera_only"',
        '"appearance_only"',
        '"event_branch_pass": None',
        '"machine_diagnostics_have_semantic_authority": False',
        'BASE_V1_SPEC_BASENAME',
        'generation._load_attempt_receipt',
    )
    if any(anchor not in adapter for anchor in adapter_anchors):
        raise SystemExit("review adapter closure differs")
    for leaked_renderer in (
        "item['branch'].upper()",
        "Candidate IDs, seeds",
    ):
        if leaked_renderer in adapter:
            raise SystemExit("review adapter retains blind identifier renderer")
    postflight_anchors = (
        '"external_postflight_after_review_terminal"',
        "validate_blind_surface(",
        "items, job_id=review_job_id, protocol_digest=protocol_digest",
        'Counter(parser.video_sources)',
        "parser.video_count != 68",
        "parser.video_end_count != 68",
        "parser.sources != parser.video_sources",
        'parser.forbidden_transport',
        '"url(" in lower',
        '"blind HTML href/src opaque namespace differs"',
        '"blind HTML leaks assessor-private identifiers or artifacts"',
        '"incomplete", "camera_only", "appearance_only"',
        'f"s{item.get(\'seed\')}" for item in items',
        '"machine_artifact_ingest_ready": True',
        '"human_label_ingest_ready": False',
        '"technical_html_visibility_to_human_observers_ready": False',
        '"review_job_terminal_success": True',
        '"review assessor-private alias mapping differs"',
        "validate_blank_template(",
        "BLANK_RESPONSE_FIELDS - {\"review_item_id\"}",
        "os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW",
        "dir_fd=parent_descriptor",
        "terminal admission same-FD reread differs",
        "sealed terminal admission identity differs",
        "validate_release_manifest(",
        "retained_plain_bytes(",
        "reread_retained(",
        'Path(__file__) != EXPECTED_POSTFLIGHT',
        '"review release manifest"',
        '"formal master/review item cross-binding differs"',
        '"formal event/source/review item cross-binding differs"',
        "EXPECTED_EVENT_SPEC_RAW_SHA256",
        "validate_formal_three_gate_bundle(submission)",
        '"formal three-gate admission binding differs"',
        '"SubmitLine%8192"',
        '"submit_line_sha256"',
        '"retained_wrapper_fd"',
        '"exact_submit_line"',
        'external_observation.get("submit_line_sha256")',
        'external_observation.get("retained_wrapper_fd")',
        'external_observation.get("exact_submit_line")',
        'expected_observer_protocol(items)',
        'formal_master_raw=formal_master_raw',
        'formal_submission_raw=formal_submission_raw',
    )
    if any(anchor not in postflight for anchor in postflight_anchors):
        raise SystemExit("review postflight closure differs")
    for anchor in (
        'os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW',
        'dir_fd=output_parent_descriptor',
        'sealed review submission receipt same-FD reread differs',
        '"release_manifest_sha256": args.release_manifest_sha256',
        '"submitter_sha256":',
        '"hostile_sha256":',
    ):
        if anchor not in submitter:
            raise SystemExit("review submitter release/publication closure differs")
    authority_sources = transport_sources
    if any(
        forbidden in authority_sources
        for forbidden in (
            '"event_verified": True',
            '"identity_preservation_verified": True',
            '"training_allowed": True',
            '"optimizer_step_allowed": True',
            '"parameter_update_allowed": True',
        )
    ):
        raise SystemExit("review release contains forbidden authority")

    bash = Path("/usr/bin/bash")
    if not bash.is_file() or bash.is_symlink() or sha256(bash) != args.expected_bash_sha256:
        raise SystemExit("AUH compute Bash identity differs")
    version = subprocess.run(
        [str(bash), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    lines = version.stdout.decode("ascii").splitlines()
    if (
        version.returncode != 0
        or version.stderr
        or hashlib.sha256(version.stdout).hexdigest()
        != args.expected_bash_version_stdout_sha256
        or not lines
        or lines[0] != args.expected_bash_version_first_line
    ):
        raise SystemExit("AUH compute Bash version contract differs")
    syntax = subprocess.run(
        [str(bash), "-n", str(launcher_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
    )
    if syntax.returncode != 0 or syntax.stdout or syntax.stderr:
        raise SystemExit("AUH Bash parser rejected review launcher")
    print("SAIC_T2V_TOPUP_REVIEW_V2_AUH_HOSTILE_STATIC_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
